"""Offline Conformer gate/padding, activation and residual fusions (expanded and hybrid stacks)."""
import contextlib
import torch
import triton
import triton.language as tl
from .fused_kernels import int8_matmul


@triton.jit
def _glu_pad(X,Mask,Y,C:tl.constexpr,T:tl.constexpr,LEFT:tl.constexpr,RIGHT:tl.constexpr,
             XB:tl.constexpr,XC:tl.constexpr,XT:tl.constexpr,HAS_MASK:tl.constexpr,BC:tl.constexpr,BT:tl.constexpr):
    b=tl.program_id(2);c=tl.program_id(0)*BC+tl.arange(0,BC)
    p=tl.program_id(1)*BT+tl.arange(0,BT);t=p-LEFT
    valid=(c[:,None]<C)&(t[None,:]>=0)&(t[None,:]<T)
    a=tl.load(X+b*XB+c[:,None]*XC+t[None,:]*XT,valid,other=0.)
    g=tl.load(X+b*XB+(c[:,None]+C)*XC+t[None,:]*XT,valid,other=0.)
    y=a*tl.sigmoid(g)
    if HAS_MASK:
        padded=tl.load(Mask+b*T+t,(t>=0)&(t<T),other=1)
        y=tl.where(padded[None,:],0.,y)
    tl.store(Y+b*C*(T+LEFT+RIGHT)+c[:,None]*(T+LEFT+RIGHT)+p[None,:],y,
             (c[:,None]<C)&(p[None,:]<T+LEFT+RIGHT))


def glu_pad(x,mask,left,right):
    b,twoc,t=x.shape;c=twoc//2
    y=torch.empty((b,c,t+left+right),dtype=x.dtype,device=x.device)
    _glu_pad[(triton.cdiv(c,32),triton.cdiv(t+left+right,32),b)](
        x,mask,y,c,t,left,right,*x.stride(),mask is not None,32,32,num_warps=4,enable_fp_fusion=False)
    return y


@triton.jit
def _residual_norm(X,R,W,B,Y,N:tl.constexpr,EPS:tl.constexpr,FACTOR:tl.constexpr,BLOCK:tl.constexpr):
    m=tl.program_id(0);n=tl.arange(0,BLOCK)
    x=tl.load(X+m*N+n,n<N,other=0)*FACTOR+tl.load(R+m*N+n,n<N,other=0)
    mean=tl.sum(x,0)/N;d=tl.where(n<N,x-mean,0.)
    inv=tl.rsqrt(tl.sum(d*d,0)/N+EPS)
    y=d*inv*tl.load(W+n,n<N,other=0)+tl.load(B+n,n<N,other=0)
    tl.store(Y+m*N+n,y,n<N)


def residual_norm(x,residual,norm,factor):
    if not x.is_contiguous() or not residual.is_contiguous():
        return norm(residual+x*factor)
    n=x.shape[-1];y=torch.empty_like(x)
    _residual_norm[(x.numel()//n,)](x,residual,norm.weight,norm.bias,y,n,norm.eps,factor,
                                   triton.next_power_of_2(n),num_warps=4,enable_fp_fusion=False)
    return y


@contextlib.contextmanager
def layer_fusions(model,project=None,*,convolution=True,residual=True):
    if model.training or torch.is_grad_enabled():raise ValueError('eval/no-grad required')
    project=project or int8_matmul;undo=[]
    def patch(obj,value):
        undo.append((obj,'forward' in obj.__dict__,obj.forward));obj.forward=value
    try:
        for layer in model.encoder.layers:
            if convolution:
                conv=layer.conv;dw=conv.depthwise_conv
                if (conv.pointwise_activation!='glu_' or conv.norm_type!='batch_norm' or dw.padding!=(0,)
                    or dw.stride!=(1,) or dw.dilation!=(1,)):
                    raise ValueError('expected folded, offline Parakeet depthwise convolution')
                original=conv.forward
                def conv_forward(x,pad_mask=None,cache=None,conv=conv,dw=dw,original=original):
                    if cache is not None:return original(x,pad_mask=pad_mask,cache=cache)
                    y=conv.pointwise_conv1(x.transpose(1,2))
                    y=glu_pad(y,pad_mask,dw._left_padding,dw._right_padding)
                    y=torch.nn.Conv1d.forward(dw,y)
                    y=conv.batch_norm(y)
                    return project(y,conv.pointwise_conv2,conv=True,components=3,input_activation='silu').transpose(1,2)
                patch(conv,conv_forward)
            if residual:
                original=layer.forward
                def forward(x,att_mask=None,pos_emb=None,pad_mask=None,cache_last_channel=None,cache_last_time=None,
                            layer=layer,original=original):
                    if (cache_last_channel is not None or cache_last_time is not None or layer.is_adapter_available()
                        or layer.is_access_enabled(getattr(layer,'model_guid',None)) or layer.self_attention_model!='rel_pos'):
                        return original(x,att_mask,pos_emb,pad_mask,cache_last_channel,cache_last_time)
                    y=layer.feed_forward1(layer.norm_feed_forward1(x))
                    r=torch.add(x,y,alpha=layer.fc_factor)
                    q=layer.norm_self_att(r)
                    r=r+layer.self_attn(query=q,key=q,value=q,mask=att_mask,pos_emb=pos_emb)
                    r=r+layer.conv(layer.norm_conv(r),pad_mask=pad_mask)
                    y=layer.feed_forward2(layer.norm_feed_forward2(r))
                    return residual_norm(y,r,layer.norm_out,layer.fc_factor)
                patch(layer,forward)
        yield
    finally:
        torch.cuda.synchronize()
        for obj,existed,original in reversed(undo):
            if existed:obj.forward=original
            else:del obj.forward
