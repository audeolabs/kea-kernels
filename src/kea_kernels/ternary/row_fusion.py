"""Row epilogue -> residual -> normalization -> INT8x3 fusion (the hybrid stack's deferred projections)."""
import contextlib
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@dataclass
class DeferredProjection:
    partial:torch.Tensor
    scale:torch.Tensor
    bias:torch.Tensor
    split:int
    shape:tuple

    def transpose(self,a,b):
        # The sole caller is the convolution adapter, which returns the
        # canonical B,T,C order after its two opposing view operations.
        if {a,b}!={1,2}:raise ValueError('unsupported deferred view')
        return self


@triton.jit
def _row_epilogue(P,S,B,R,Y,Q,QS,NW,NB,OW,OB,
                  M:tl.constexpr,N:tl.constexpr,SPLIT:tl.constexpr,FACTOR:tl.constexpr,HAS_BIAS:tl.constexpr,
                  QUANTIZE:tl.constexpr,OUT_NORM:tl.constexpr,EPS:tl.constexpr,O_EPS:tl.constexpr,
                  BLOCK:tl.constexpr):
    row=tl.program_id(0);n=tl.arange(0,BLOCK)
    value=tl.full((BLOCK,),0,tl.float32)
    for part in tl.static_range(SPLIT):
        value+=tl.load(P+part*M*N+row*N+n,n<N,other=0)
    if SPLIT>1:
        if HAS_BIAS:value=tl.fma(value,tl.load(S+n,n<N,other=0),tl.load(B+n,n<N,other=0))
        else:value*=tl.load(S+n,n<N,other=0)
    x=tl.load(R+row*N+n,n<N,other=0)+value*FACTOR
    if OUT_NORM:
        mean=tl.sum(x,0)/N;d=tl.where(n<N,x-mean,0.)
        x=d*tl.rsqrt(tl.sum(d*d,0)/N+O_EPS)*tl.load(OW+n,n<N,other=0)+tl.load(OB+n,n<N,other=0)
    tl.store(Y+row*N+n,x,n<N)
    if QUANTIZE:
        mean=tl.sum(x,0)/N;d=tl.where(n<N,x-mean,0.)
        x=d*tl.rsqrt(tl.sum(d*d,0)/N+EPS)*tl.load(NW+n,n<N,other=0)+tl.load(NB+n,n<N,other=0)
        for part in tl.static_range(3):
            scale=tl.maximum(tl.max(tl.abs(x),0)/127.,1.e-30)
            q=tl.extra.cuda.libdevice.nearbyint(x/scale).to(tl.int8)
            tl.store(Q+part*M*N+row*N+n,q,n<N)
            tl.store(QS+part*M+row,scale)
            x=x-q.to(tl.float32)*scale


@gluon.jit
def _row_epilogue_pinned(P,S,B,R,Y,Q,QS,NW,NB,OW,OB,
                  M:gl.constexpr,N:gl.constexpr,SPLIT:gl.constexpr,FACTOR:gl.constexpr,HAS_BIAS:gl.constexpr,
                  QUANTIZE:gl.constexpr,OUT_NORM:gl.constexpr,EPS:gl.constexpr,O_EPS:gl.constexpr,
                  BLOCK:gl.constexpr):
    # Pin the original layouts; LLVM can still choose a different local
    # FP32 reduction tree. This variant therefore also needs ASR validation.
    layout:gl.constexpr=gl.BlockedLayout([8],[32],[4],[0])
    out_layout:gl.constexpr=gl.BlockedLayout([4],[32],[4],[0])
    row=gl.program_id(0);n=gl.arange(0,BLOCK,layout=layout)
    value=gl.full((BLOCK,),0,gl.float32,layout=layout)
    for part in gl.static_range(SPLIT):value+=gl.load(P+part*M*N+row*N+n,n<N,other=0)
    if SPLIT>1:
        if HAS_BIAS:value=gl.fma(value,gl.load(S+n,n<N,other=0),gl.load(B+n,n<N,other=0))
        else:value*=gl.load(S+n,n<N,other=0)
    x=gl.load(R+row*N+n,n<N,other=0)+value*FACTOR
    if OUT_NORM:
        x=gl.convert_layout(x,out_layout)
        nn=gl.arange(0,BLOCK,layout=out_layout)
        mean=gl.sum(x,0)/N;d=gl.where(nn<N,x-mean,0.)
        x=d*gl.rsqrt(gl.sum(d*d,0)/N+O_EPS)*gl.load(OW+nn,nn<N,other=0)+gl.load(OB+nn,nn<N,other=0)
        gl.store(Y+row*N+nn,x,nn<N)
        x=gl.convert_layout(x,layout)
    else:gl.store(Y+row*N+n,x,n<N)
    if QUANTIZE:
        mean=gl.sum(x,0)/N;d=gl.where(n<N,x-mean,0.)
        x=d*gl.rsqrt(gl.sum(d*d,0)/N+EPS)*gl.load(NW+n,n<N,other=0)+gl.load(NB+n,n<N,other=0)
        for part in gl.static_range(3):
            scale=gl.maximum(gl.max(gl.abs(x),0)/127.,1.e-30)
            q=gl.extra.libdevice.nearbyint(x/scale).to(gl.int8)
            gl.store(Q+part*M*N+row*N+n,q,n<N)
            gl.store(QS+part*M+row,scale)
            x=x-q.to(gl.float32)*scale


def finish_residual(provider,value,residual,factor,norm=None,out_norm=None):
    if not isinstance(value,DeferredProjection):
        raise TypeError('row fusion requires a deferred CUDA projection')
    if not residual.is_contiguous() or residual.shape!=value.shape:
        raise ValueError('expected contiguous B,T,C residual')
    n=residual.shape[-1];m=residual.numel()//n
    y=torch.empty_like(residual)
    q=torch.empty(3,m,n,device=y.device,dtype=torch.int8) if norm is not None else provider.null
    qs=torch.empty(3,m,device=y.device) if norm is not None else provider.null
    warps=getattr(provider,'row_warps',4)
    if warps=='adaptive':warps=8 if m<=192 else 4
    _row_epilogue[(m,)](value.partial,value.scale,value.bias,residual,y,q,qs,
         norm.weight if norm is not None else None,norm.bias if norm is not None else None,
         out_norm.weight if out_norm is not None else None,out_norm.bias if out_norm is not None else None,
         m,n,value.split,factor,value.bias is not None,norm is not None,out_norm is not None,
         norm.eps if norm is not None else 0.,out_norm.eps if out_norm is not None else 0.,
         triton.next_power_of_2(n),num_warps=warps,enable_fp_fusion=False)
    if norm is not None:
        key=(y.data_ptr(),m,n,id(norm))
        if key in provider.prequantized:raise RuntimeError('unconsumed fused activation')
        provider.prequantized[key]=(q,qs)
    return y


@contextlib.contextmanager
def deferred(provider):
    previous=provider.defer_epilogue;provider.defer_epilogue=True
    try:yield
    finally:provider.defer_epilogue=previous


@contextlib.contextmanager
def row_fusions(model,provider):
    undo=[]
    provider.no_defer_weights={layer.self_attn.linear_pos.packed_t.data_ptr() for layer in model.encoder.layers}
    try:
        layers=model.encoder.layers
        for i,layer in enumerate(layers):
            original=layer.forward
            following=layers[i+1].norm_feed_forward1 if i+1<len(layers) else None
            def forward(x,att_mask=None,pos_emb=None,pad_mask=None,cache_last_channel=None,cache_last_time=None,
                        layer=layer,following=following):
                if cache_last_channel is not None or cache_last_time is not None:
                    raise ValueError('row-fusion experiment is offline only')
                with deferred(provider):value=layer.feed_forward1(layer.norm_feed_forward1(x))
                r=finish_residual(provider,value,x,layer.fc_factor,layer.norm_self_att)
                with deferred(provider):
                    value=layer.self_attn(query=r,key=r,value=r,mask=att_mask,pos_emb=pos_emb)
                r=finish_residual(provider,value,r,1.,layer.norm_conv)
                with deferred(provider):value=layer.conv(r,pad_mask=pad_mask)
                r=finish_residual(provider,value,r,1.,layer.norm_feed_forward2)
                with deferred(provider):value=layer.feed_forward2(r)
                return finish_residual(provider,value,r,layer.fc_factor,following,layer.norm_out)
            undo.append((layer,'forward' in layer.__dict__,original));layer.forward=forward
        yield
        if provider.prequantized:raise RuntimeError('fused activation cache not fully consumed')
    finally:
        torch.cuda.synchronize()
        for layer,existed,original in reversed(undo):
            if existed:layer.forward=original
            else:del layer.forward
        provider.prequantized.clear()
        provider.no_defer_weights.clear()
