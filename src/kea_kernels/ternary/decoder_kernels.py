"""FP32 TDT predictor/joint fusions for the validated SM120 inference path."""
import contextlib
import torch
import triton
import triton.language as tl


def _retain_for_graph(computer, *tensors):
    # Own raw Triton arguments for the graph lifetime. This alone cannot
    # protect temporary allocations inside PyTorch/library operations:
    # decoder_graph_pool also covers the conditional child streams.
    if torch.cuda.is_current_stream_capturing():
        state = computer.state
        if not hasattr(state, '_packed_decoder_keepalive'):
            state._packed_decoder_keepalive = []
        state._packed_decoder_keepalive.extend(t for t in tensors if t is not None)


@triton.jit
def _linear(X, Z, W, Bias, Y, K:tl.constexpr, N:tl.constexpr,
            BN:tl.constexpr, BK:tl.constexpr, ADD_RELU:tl.constexpr):
    b=tl.program_id(1)
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    x=tl.load(X+b*K+k,k<K,other=0)
    if ADD_RELU:
        x=tl.maximum(x+tl.load(Z+b*K+k,k<K,other=0),0.)
    w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),other=0)
    y=tl.sum(w*x[None,:],1)+tl.load(Bias+n,n<N,other=0)
    tl.store(Y+b*N+n,y,n<N)


def linear(x,mod,z=None):
    n,k=mod.weight.shape
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    _linear[(triton.cdiv(n,4),x.numel()//k)](x,z,mod.weight,mod.bias,y,k,n,4,triton.next_power_of_2(k),z is not None,
                                           num_warps=4,enable_fp_fusion=False)
    return y


@triton.jit
def _lstm_step(X, H, C, WI, WH, BI, BH, TABLE, Tokens, HO, CO,
               D:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr, LOOKUP:tl.constexpr):
    b=tl.program_id(1)
    idx=tl.program_id(0)*BN+tl.arange(0,BN)
    row=tl.arange(0,4*BN)
    n=(row//BN)*D+tl.program_id(0)*BN+(row%BN)
    k=tl.arange(0,BK)
    mask=((tl.program_id(0)*BN+row[:,None]%BN)<D)&(k[None,:]<D)
    h=tl.load(H+b*D+k,k<D,other=0)
    wh=tl.load(WH+n[:,None]*D+k[None,:],mask,other=0)
    recurrent=tl.sum(wh*h[None,:],1)+tl.load(BH+n,(tl.program_id(0)*BN+row%BN)<D,other=0)
    if LOOKUP:
        token=tl.load(Tokens+b)
        inp=tl.load(TABLE+token*4*D+n,(tl.program_id(0)*BN+row%BN)<D,other=0)
    else:
        x=tl.load(X+b*D+k,k<D,other=0)
        wi=tl.load(WI+n[:,None]*D+k[None,:],mask,other=0)
        inp=tl.sum(wi*x[None,:],1)
    gates=(inp+tl.load(BI+n,(tl.program_id(0)*BN+row%BN)<D,other=0)+recurrent).reshape(4,BN)
    gi=tl.sum(tl.where(tl.arange(0,4)[:,None]==0,gates,0.),0)
    gf=tl.sum(tl.where(tl.arange(0,4)[:,None]==1,gates,0.),0)
    gg=tl.sum(tl.where(tl.arange(0,4)[:,None]==2,gates,0.),0)
    go=tl.sum(tl.where(tl.arange(0,4)[:,None]==3,gates,0.),0)
    c=tl.load(C+b*D+idx,idx<D,other=0)
    newc=tl.sigmoid(gf)*c+tl.sigmoid(gi)*tl.extra.cuda.libdevice.tanh(gg)
    newh=tl.sigmoid(go)*tl.extra.cuda.libdevice.tanh(newc)
    tl.store(CO+b*D+idx,newc,idx<D);tl.store(HO+b*D+idx,newh,idx<D)


@contextlib.contextmanager
def decoder_transform(model, *, joint=True, lstm=False, precompute=False, block=2):
    """Reversible FP32 inference transform; preserves NeMo's decoding algorithm.

    Changes require graph reset because captured decoder graphs retain method
    implementations. Keep the context alive until all graph work completes.
    """
    if model.training or torch.is_grad_enabled():raise ValueError('eval/no-grad required')
    dec,j=model.decoder,model.joint
    if dec.is_adapter_available() or j.is_adapter_available():raise ValueError('decoder adapters unsupported')
    comp=model.decoding.decoding.decoding_computer
    comp.reset_cuda_graphs_state()
    undo=[]
    def patch(obj,name,value):
        undo.append((obj,name,name in obj.__dict__,getattr(obj,name)))
        setattr(obj,name,value)
    try:
        if joint:
            original_pred=j.project_prednet
            def project(x):
                if x.dtype!=torch.float32 or not x.is_contiguous() or x.numel()//x.shape[-1]>8:return original_pred(x)
                y=linear(x,j.pred);_retain_for_graph(comp,x,y)
                return y
            patch(j,'project_prednet',project)
            original_joint=j.joint_after_projection
            def join(f,g):
                if (f.dtype!=torch.float32 or f.shape[1]!=1 or g.shape[1]!=1 or not f.is_contiguous()
                    or not g.is_contiguous() or f.shape[0]>8 or j.log_softmax or not isinstance(j.joint_net[0],torch.nn.ReLU)):
                    return original_joint(f,g)
                y=linear(f,j.joint_net[-1],z=g);_retain_for_graph(comp,f,g,y)
                return y.unsqueeze(2)
            patch(j,'joint_after_projection',join)
        if lstm:
            rnn=dec.prediction['dec_rnn'].lstm
            if rnn.num_layers!=2 or rnn.hidden_size!=640 or rnn.bidirectional or rnn.proj_size:
                raise ValueError('expected two-layer 640-unit LSTM')
            tf32=torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32=False
                table=(dec.prediction['embed'].weight @ rnn.weight_ih_l0.T).contiguous() if precompute else None
            finally:
                torch.backends.cuda.matmul.allow_tf32=tf32
            original=dec.predict
            def predict(y=None,state=None,add_sos=True,batch_size=None):
                if (y is None or state is None or add_sos or y.ndim!=2 or y.shape[1]!=1 or y.shape[0]>8
                    or state[0].dtype!=torch.float32 or not y.is_contiguous() or not all(v.is_contiguous() for v in state)):
                    return original(y,state,add_sos,batch_size)
                h,c=state
                ho,co=torch.empty_like(h),torch.empty_like(c)
                x=None if precompute else dec.prediction['embed'](y)
                _retain_for_graph(comp,y,h,c,ho,co,x)
                for layer in range(2):
                    _lstm_step[(triton.cdiv(640,block),y.shape[0])](
                        x,h[layer],c[layer],getattr(rnn,f'weight_ih_l{layer}'),getattr(rnn,f'weight_hh_l{layer}'),
                        getattr(rnn,f'bias_ih_l{layer}'),getattr(rnn,f'bias_hh_l{layer}'),table,y,ho[layer],co[layer],
                        640,block,1024,precompute and layer==0,num_warps=4,enable_fp_fusion=False)
                    x=ho[layer]
                return ho[-1].unsqueeze(1),(ho,co)
            patch(dec,'predict',predict)
        yield
    finally:
        torch.cuda.synchronize()
        comp.reset_cuda_graphs_state()
        for obj,name,existed,old in reversed(undo):
            if existed:setattr(obj,name,old)
            else:delattr(obj,name)


@triton.jit
def _indexed_joint(E,G,Time,W,Bias,Y,T:tl.constexpr,K:tl.constexpr,N:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
    t=tl.load(Time);t=tl.where(t<0,t+T,t)
    x=tl.maximum(tl.load(E+t*K+k,k<K,other=0)+tl.load(G+k,k<K,other=0),0.)
    w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),other=0)
    y=tl.sum(w*x[None,:],1)+tl.load(Bias+n,n<N,other=0)
    tl.store(Y+n,y,n<N)


@triton.jit
def _advance(Logits,Labels,Scores,Durations,ModelDurations,Time,Safe,Current,Last,Length,
             Active,Prev,Blank,Advance,Any,BLANK:tl.constexpr,ND:tl.constexpr,INNER:tl.constexpr,V:tl.constexpr):
    v=tl.arange(0,V)
    logits=tl.load(Logits+v,v<=BLANK,other=-float('inf'))
    label=tl.argmax(logits,0);score=tl.max(logits,0)
    d=tl.arange(0,8)
    dl=tl.load(Logits+BLANK+1+d,d<ND,other=-float('inf'))
    duration=tl.load(ModelDurations+tl.argmax(dl,0))
    t=tl.load(Time);active=tl.load(Active)
    if INNER:
        advance=tl.load(Advance)
        label=tl.where(advance,label,tl.load(Labels));score=tl.where(advance,score,tl.load(Scores))
        tl.store(Current,tl.where(advance,t,tl.load(Current)))
    else:
        tl.store(Prev,active);tl.store(Current,t)
        advance=active
    blank=label==BLANK
    duration=tl.where(blank&(duration==0),1,duration)
    t=t+tl.where(advance,duration,0)
    if INNER: duration=tl.where(advance,duration,tl.load(Durations))
    active=t<tl.load(Length)
    tl.store(Labels,label);tl.store(Scores,score);tl.store(Durations,duration)
    tl.store(Time,t);tl.store(Safe,tl.minimum(t,tl.load(Last)))
    tl.store(Active,active);tl.store(Blank,blank)
    advance=active&blank
    tl.store(Advance,advance);tl.store(Any,advance)


@triton.jit
def _force_symbols(Labels,Time,Safe,Last,Length,Active,Any,Stamp,Count,BLANK:tl.constexpr,MAX:tl.constexpr):
    t=tl.load(Time)
    force=tl.load(Active)&(tl.load(Labels)!=BLANK)&(tl.load(Count)>=MAX)&(tl.load(Stamp)==t)
    t+=force.to(t.dtype)
    active=t<tl.load(Length)
    tl.store(Time,t);tl.store(Safe,tl.minimum(t,tl.load(Last)));tl.store(Active,active);tl.store(Any,active)


@contextlib.contextmanager
def control_transform(model, *, storage=False):
    """Fuse scalar bookkeeping inside NeMo's batch-one conditional decoder.

    Larger batches, fusion models, confidence/logit recording use original code.
    With storage=True, token storage and recurrent-state selection are fused too.
    """
    comp=model.decoding.decoding.decoding_computer
    if model.training or torch.is_grad_enabled():raise ValueError('eval/no-grad required')
    comp.reset_cuda_graphs_state();undo=[]
    def patch(name,value):
        undo.append((name,name in comp.__dict__,getattr(comp,name)));setattr(comp,name,value)
    def eligible():
        s=comp.state
        return (s.batch_size==1 and s.float_dtype==torch.float32 and not comp.has_fusion_models()
                and not comp.record_all_steps and not comp.preserve_logits and not comp.preserve_step_confidence
                and not model.joint.log_softmax and s.model_durations.numel()<=8
                and isinstance(comp.max_symbols,int) and comp.max_symbols>0
                and isinstance(model.joint.joint_net[0],torch.nn.ReLU) and not model.joint.is_adapter_available())
    def step(inner):
        s=comp.state;mod=model.joint.joint_net[-1];n,k=mod.weight.shape
        logits=torch.empty((n,),device=mod.weight.device,dtype=torch.float32)
        _retain_for_graph(comp,logits)
        _indexed_joint[(triton.cdiv(n,4),)](s.encoder_output_projected,s.decoder_output,s.safe_time_indices,
                   mod.weight,mod.bias,logits,s.max_time,k,n,4,triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
        _advance[(1,)](logits,s.labels,s.scores,s.durations,s.model_durations,s.time_indices,s.safe_time_indices,
                 s.time_indices_current_labels,s.last_timestamps,s.encoder_output_length,s.active_mask,s.active_mask_prev,
                 s.blank_mask,s.advance_mask,s.advance_mask_any,comp._blank_index,s.model_durations.numel(),inner,
                 triton.next_power_of_2(comp._blank_index+1),num_warps=4)
    try:
        before=comp._before_inner_loop_get_joint_output
        inner=comp._inner_loop_step_find_next_non_blank
        force=comp._after_inner_loop_force_max_symbols
        patch('_before_inner_loop_get_joint_output',lambda:step(False) if eligible() else before())
        patch('_inner_loop_step_find_next_non_blank',lambda:step(True) if eligible() else inner())
        def force_symbols():
            if not eligible():return force()
            s=comp.state;h=s.batched_hyps
            _force_symbols[(1,)](s.labels,s.time_indices,s.safe_time_indices,s.last_timestamps,s.encoder_output_length,
                      s.active_mask,s.active_mask_any,h.last_nb_timestamp,h.last_nb_timestamp_lasts,comp._blank_index,
                      comp.max_symbols,num_warps=1)
        patch('_after_inner_loop_force_max_symbols',force_symbols)
        if storage:
            original_store=comp._after_inner_loop_store_labels
            original_predict=comp._after_inner_loop_get_decoder_output
            def store_labels():
                if not eligible():return original_store()
                s=comp.state;h=s.batched_hyps
                _store_label[(1,)](s.active_mask_prev,s.labels,s.time_indices_current_labels,s.scores,s.durations,
                        s.found_labels_mask,h.current_lengths,h.transcript,h.timestamps,h.token_durations,
                        h.last_nb_timestamp,h.last_nb_timestamp_lasts,h.last_nb_labels,h.scores,comp._blank_index,
                        comp.include_duration,num_warps=1)
            def update_predictor():
                if not eligible():return original_predict()
                s=comp.state
                output,new_state=model.decoder.predict(s.labels.unsqueeze(1),s.decoder_state,add_sos=False,batch_size=1)
                projected=model.joint.project_prednet(output)
                h,c=new_state;dh,dc=s.decoder_state
                _retain_for_graph(comp,h,c,projected)
                _select_state[(triton.cdiv(h.numel(),256),)](h,c,projected,dh,dc,s.decoder_output,s.found_labels_mask,
                        h.numel(),projected.numel(),256,num_warps=4)
            patch('_after_inner_loop_store_labels',store_labels)
            patch('_after_inner_loop_get_decoder_output',update_predictor)
        yield
    finally:
        torch.cuda.synchronize();comp.reset_cuda_graphs_state()
        for name,existed,old in reversed(undo):
            if existed:setattr(comp,name,old)
            else:delattr(comp,name)


@triton.jit
def _store_label(Prev,Labels,Current,Scores,Durations,Found,Lengths,Transcript,Timestamps,TokenDurations,
                  Last,Count,LastLabel,Total,BLANK:tl.constexpr,WITH_DURATION:tl.constexpr):
    label=tl.load(Labels);time=tl.load(Current);found=tl.load(Prev)&(label!=BLANK)
    length=tl.load(Lengths)
    # NeMo writes the next slot even for an inactive hypothesis, then leaves
    # its logical length unchanged. Preserve that observable storage behavior.
    tl.store(Transcript+length,label);tl.store(Timestamps+length,time)
    if WITH_DURATION:tl.store(TokenDurations+length,tl.load(Durations))
    last=tl.load(Last);count=tl.load(Count)
    count=tl.where(found,tl.where(last==time,count+1,1),count)
    tl.store(Last,tl.where(found,time,last));tl.store(Count,count)
    tl.store(LastLabel,tl.where(found,label,tl.load(LastLabel)))
    tl.store(Total,tl.where(found,tl.load(Total)+tl.load(Scores),tl.load(Total)))
    tl.store(Lengths,length+found.to(length.dtype));tl.store(Found,found)


@triton.jit
def _select_state(H,C,P,DH,DC,DP,Mask,HN:tl.constexpr,PN:tl.constexpr,BLOCK:tl.constexpr):
    n=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=tl.load(Mask)
    h=tl.load(H+n,n<HN,other=0);c=tl.load(C+n,n<HN,other=0);p=tl.load(P+n,n<PN,other=0)
    tl.store(DH+n,h,(n<HN)&mask);tl.store(DC+n,c,(n<HN)&mask);tl.store(DP+n,p,(n<PN)&mask)
