"""The expanded projection: an exact INT8 copy of the ternary codes in column-major tiles tuned on the RTX 5090."""
import torch
import triton
import triton.language as tl
from .fused_kernels import _quantize, _int8_dot, integer_configuration
from .kernels import _finish


def expanded_configuration(m,n,k):
    """5090 column-major tiles screened at 38, 126 and 376 encoder frames."""
    band=0 if m<=64 else 1 if m<=192 else 2
    table={
        (2048,1024):[(32,64,128,4,4,2),(16,64,128,1,4,2),(32,64,128,1,4,2)],
        (1024,1024):[(32,64,256,4,4,2),(32,64,256,4,4,2),(32,64,128,1,4,2)],
        (4096,1024):[(16,64,128,1,4,2),(64,64,128,1,4,2),(32,128,128,1,4,2)],
        (1024,4096):[(64,64,128,8,4,2),(64,64,128,8,4,2),(32,128,128,8,4,2)],
        (3072,1024):[(16,64,128,1,4,2),(32,64,128,1,4,2),(64,64,128,1,4,2)],
    }
    if m>512 or (n,k) not in table:return integer_configuration(m,n,k)
    return table[n,k][band]


@triton.jit
def _stacked_dot(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.int32)
    for start in range(tl.cdiv(K,BK)):
        k=start*BK+kk
        a=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),other=0)
        b=tl.load(W+n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),other=0)
        acc=tl.dot(a,b,acc,out_dtype=tl.int32)
    tl.store(Y+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))


@triton.jit
def _float_quantize(X,Q,S,NW,NB,M:tl.constexpr,K:tl.constexpr,T:tl.constexpr,XB:tl.constexpr,XT:tl.constexpr,
                    XK:tl.constexpr,BLOCK:tl.constexpr,PARTS:tl.constexpr,NORMALIZE:tl.constexpr,EPS:tl.constexpr,
                    RANGE:tl.constexpr):
    m=tl.program_id(0);k=tl.arange(0,BLOCK)
    x=tl.load(X+(m//T)*XB+(m%T)*XT+k*XK,k<K,other=0)
    if NORMALIZE:
        mean=tl.sum(x,0)/K;d=tl.where(k<K,x-mean,0.)
        x=d*tl.rsqrt(tl.sum(d*d,0)/K+EPS)*tl.load(NW+k,k<K,other=0)+tl.load(NB+k,k<K,other=0)
    for part in tl.static_range(PARTS):
        scale=tl.maximum(tl.max(tl.abs(x),0)/RANGE,1.e-30)
        q=(x/scale).to(Q.dtype.element_ty)
        tl.store(Q+part*M*K+m*K+k,q,k<K);tl.store(S+part*M+m,scale)
        x-=q.to(tl.float32)*scale


@triton.jit
def _expand(W,Y,K:tl.constexpr,N:tl.constexpr,COLUMN:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    if COLUMN:
        n=i//K;k=i%K
    else:
        k=i//N;n=i%N
    word=tl.load(W+(k//16)*N+n,i<K*N,other=0).to(tl.uint32)
    c=(word>>(2*(k%16)))&3
    value=(c&1).to(tl.int32)-(c>>1).to(tl.int32)
    tl.store(Y+i,value.to(tl.float32).to(Y.dtype.element_ty),i<K*N)


@triton.jit
def _combine(C,S,W,B,Y,M:tl.constexpr,N:tl.constexpr,PARTS:tl.constexpr,BIAS:tl.constexpr,
             ACT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);m=i//N;n=i%N
    value=tl.load(C+i,i<M*N,other=0).to(tl.float32)*tl.load(S+m,m<M,other=0)
    for part in tl.static_range(1,PARTS):
        value+=tl.load(C+part*M*N+i,i<M*N,other=0).to(tl.float32)*tl.load(S+part*M+m,m<M,other=0)
    value*=tl.load(W+n,n<N,other=0)
    if BIAS:value+=tl.load(B+n,n<N,other=0)
    if ACT=='silu':value=value/(1.+tl.exp(-value))
    tl.store(Y+i,value,i<M*N)


class ExpandedProjection:
    """Own exact decoded INT8 weights until all caller graphs have been freed."""
    def __init__(self,backend='triton',config=None):
        pieces=backend.split(':',1)
        self.backend=pieces[0];self.families=set(pieces[1].split(',')) if len(pieces)>1 else None
        self.weights={};self.config=config

    @property
    def bytes(self):return sum(w.numel()*w.element_size() for w in self.weights.values())

    def __call__(self,x,mod,conv=False,components=3,activation='',norm=None,config=None,input_activation=''):
        xx=x.transpose(1,2) if conv else x
        if xx.ndim==2:xx=xx.unsqueeze(0)
        b,t,k=xx.shape;m=b*t;n=mod.out_features
        family='down' if k==4096 else {4096:'up',3072:'qkv',2048:'gate',1024:'mix'}.get(n,'other')
        if self.families is not None and family not in self.families:
            from .fused_kernels import int8_matmul
            return int8_matmul(x,mod,conv,components,activation,norm,config,input_activation)
        if self.backend.startswith('torch') and (k%8 or n%8 or components*m<=16):
            from .fused_kernels import int8_matmul
            return int8_matmul(x,mod,conv,components,activation,norm,config,input_activation)
        floating=self.backend in ('fp16x2','bf16x3','fp8x3','fp8x4')
        stacked=self.backend.startswith('stacked')
        column=self.backend in ('torch_col','triton_col','triton_col_tuned') or floating or stacked
        if floating:components={'fp16x2':2,'bf16x3':3,'fp8x3':3,'fp8x4':4}[self.backend]
        dtype=({'fp16x2':torch.float16,'bf16x3':torch.bfloat16,'fp8x3':torch.float8_e4m3fn,
                'fp8x4':torch.float8_e4m3fn}[self.backend] if floating else torch.int8)
        key=(mod.packed_t.data_ptr(),k,n,column)
        if key not in self.weights:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('warm expanded weights before capture')
            w=torch.empty((n,k) if column else (k,n),dtype=dtype,device=x.device)
            _expand[(triton.cdiv(k*n,512),)](mod.packed_t,w,k,n,column,512)
            self.weights[key]=w.T if column else w
        w=self.weights[key]
        q=torch.empty((components,m,k),dtype=dtype,device=x.device)
        scales=torch.empty((components,m),device=x.device)
        y=torch.empty((b,t,n),device=x.device)
        if floating:
            if input_activation:raise ValueError('input activation fusion requires integer backend')
            _float_quantize[(m,)](xx,q,scales,norm.weight if norm is not None else None,norm.bias if norm is not None else None,
                           m,k,t,*xx.stride(),triton.next_power_of_2(k),components,norm is not None,
                           norm.eps if norm is not None else 0.,256. if self.backend.startswith('fp8') else 512.,enable_fp_fusion=False)
        else:
            _quantize[(m,)](xx,q,scales,norm.weight if norm is not None else None,norm.bias if norm is not None else None,
                           m,k,t,*xx.stride(),triton.next_power_of_2(k),components,norm is not None,
                           norm.eps if norm is not None else 0.,INPUT_SILU=input_activation=='silu',enable_fp_fusion=False)
        if self.backend.startswith('torch') or floating or stacked:
            if self.backend.startswith('fp8'):
                if not hasattr(self,'one'):self.one=torch.ones((),device=x.device)
                accum=torch._scaled_mm(q.reshape(components*m,k),w,scale_a=self.one,scale_b=self.one,
                                      out_dtype=torch.float32,use_fast_accum=False)
            elif floating:accum=torch.mm(q.reshape(components*m,k),w,out_dtype=torch.float32)
            elif stacked:
                bm,bn,bk,warps,stages={
                    'stacked':(64,64,64,4,3),'stacked_wide':(64,128,128,8,3),
                    'stacked_large':(128,128,64,8,3),'stacked_deep':(64,64,128,4,4)}[self.backend]
                accum=torch.empty((components*m,n),device=x.device,dtype=torch.int32)
                _stacked_dot[(triton.cdiv(components*m,bm),triton.cdiv(n,bn))](q,w,accum,components*m,n,k,bm,bn,bk,
                                                                          num_warps=warps,num_stages=stages)
            else:accum=torch._int_mm(q.reshape(components*m,k),w)
            _combine[(triton.cdiv(m*n,256),)](accum,scales,mod.scale,mod.bias,y,m,n,components,mod.bias is not None,
                                           activation,256,enable_fp_fusion=False)
        else:
            configuration=expanded_configuration if self.backend=='triton_col_tuned' else integer_configuration
            bm,bn,bk,split,warps,stages=self.config or config or configuration(m,n,k)
            partial=torch.empty((split,m,n),device=x.device) if split>1 else y
            _int8_dot[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](
                q,scales,w,mod.scale,mod.bias,y,partial,m,n,k,mod.bias is not None,bm,bn,bk,split,components,
                ACTIVATION=activation,DENSE=True,COLUMN=column,num_warps=warps,num_stages=stages)
            if split>1:
                _finish[(triton.cdiv(m*n,256),)](partial,mod.scale,mod.bias,y,m,n,t,t*n,n,1,mod.bias is not None,
                                              split,256,ACTIVATION=activation)
        return y.transpose(1,2) if conv else y.reshape(*x.shape[:-1],n)
