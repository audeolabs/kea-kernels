"""SM120 residual-integer projections, normalization, and relative attention."""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, W, B, Y, N: tl.constexpr, ROW: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, BLOCK)
    x = tl.load(X + row * ROW + n, n < N, other=0)
    mean = tl.sum(x, 0) / N
    d = tl.where(n < N, x - mean, 0)
    var = tl.sum(d * d, 0) / N
    w = tl.load(W + n, n < N, other=0)
    b = tl.load(B + n, n < N, other=0)
    y = d * tl.rsqrt(var + EPS) * w + b
    tl.store(Y + row * N + n, y, n < N)


def layer_norm(x, norm):
    if not x.is_contiguous() or x.ndim < 2 or x.dtype != torch.float32:
        return torch.nn.functional.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    n = x.shape[-1]
    out = torch.empty_like(x)
    _norm[(x.numel() // n,)](x, norm.weight, norm.bias, out, n, n, norm.eps,
                            triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _quantize(X, Q, S, NW, NB, M: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
              XB: tl.constexpr, XT: tl.constexpr, XK: tl.constexpr, BLOCK: tl.constexpr,
              COMPONENTS: tl.constexpr, NORMALIZE: tl.constexpr, EPS: tl.constexpr, INPUT_SILU: tl.constexpr = False):
    m = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + (m // T) * XB + (m % T) * XT + k * XK, k < K, other=0)
    if NORMALIZE:
        mean = tl.sum(x, 0) / K
        centered = tl.where(k < K, x-mean, 0.)
        var = tl.sum(centered*centered, 0) / K
        x = centered * tl.rsqrt(var+EPS) * tl.load(NW+k, k < K, other=0) + tl.load(NB+k, k < K, other=0)
    if INPUT_SILU:
        x = x * tl.sigmoid(x)
    for part in tl.static_range(COMPONENTS):
        scale = tl.maximum(tl.max(tl.abs(x), 0) / 127.0, 1.e-30)
        q = tl.extra.cuda.libdevice.nearbyint(x / scale).to(tl.int8)
        tl.store(Q + part * M * K + m * K + k, q, k < K)
        tl.store(S + part * M + m, scale)
        x = x - q.to(tl.float32) * scale


@triton.jit
def _int8_dot(X, XS, W, WS, Bias, Y, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              SPLIT: tl.constexpr, COMPONENTS: tl.constexpr, ACTIVATION: tl.constexpr = "", DENSE: tl.constexpr = False,
              COLUMN: tl.constexpr = False, WEIGHT_BITS: tl.constexpr = 8, CTA_GROUPS: tl.constexpr = 1):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    k = tl.arange(0, BK)
    words = tl.arange(0, BK // 16)
    lane = tl.arange(0, 16)
    acc = tl.zeros((BM, BN), tl.int32)
    acc2 = tl.zeros((BM, BN), tl.int32)
    acc3 = tl.zeros((BM, BN), tl.int32)
    for tile in range(tl.cdiv(K, BK * SPLIT * CTA_GROUPS)):
        start = (tile * SPLIT * CTA_GROUPS + part) * BK
        a = tl.load(X + m[:, None] * K + (start+k[None, :]),
                    (m[:, None] < M) & (start+k[None, :] < K), other=0)
        if DENSE:
            if COLUMN:
                if WEIGHT_BITS == 8:
                    b = tl.load(W+n[None,:]*K+start+k[:,None],(start+k[:,None]<K)&(n[None,:]<N),other=0)
                else:
                    pk=tl.arange(0,BK//(8//WEIGHT_BITS))
                    packed=tl.load(W+n[:,None]*(K//(8//WEIGHT_BITS))+start//(8//WEIGHT_BITS)+pk[None,:],
                                   (n[:,None]<N)&(start//(8//WEIGHT_BITS)+pk[None,:]<K//(8//WEIGHT_BITS)),other=0)
                    if WEIGHT_BITS == 4:
                        low=(packed<<4).to(tl.int8)>>4
                        high=packed.to(tl.int8)>>4
                        b=tl.join(low,high).reshape(BN,BK).trans()
                    else:
                        c0=packed&3;c1=(packed>>2)&3;c2=(packed>>4)&3;c3=(packed>>6)&3
                        v0=((c0&1).to(tl.int8)-(c0>>1).to(tl.int8))
                        v1=((c1&1).to(tl.int8)-(c1>>1).to(tl.int8))
                        v2=((c2&1).to(tl.int8)-(c2>>1).to(tl.int8))
                        v3=((c3&1).to(tl.int8)-(c3>>1).to(tl.int8))
                        b=tl.join(tl.join(v0,v1),tl.join(v2,v3)).permute(0,1,3,2).reshape(BN,BK).trans()
            else:
                b = tl.load(W+(start+k[:,None])*N+n[None,:],(start+k[:,None]<K)&(n[None,:]<N),other=0)
        else:
            w = tl.load(W + (start // 16 + words[:, None]) * N + n[None, :],
                        (start // 16 + words[:, None] < tl.cdiv(K, 16)) & (n[None, :] < N), other=0)
            c = ((w[:, None, :].to(tl.uint32) >> (lane[None, :, None] * 2)) & 3).reshape(BK, BN)
            b = ((c & 1).to(tl.int32) - (c >> 1).to(tl.int32)).to(tl.int8)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        if COMPONENTS > 1:
            a2 = tl.load(X + M*K + m[:, None]*K + start+k[None, :],
                         (m[:, None] < M) & (start+k[None, :] < K), other=0)
            acc2 = tl.dot(a2, b, acc2, out_dtype=tl.int32)
        if COMPONENTS > 2:
            a3 = tl.load(X + 2*M*K + m[:, None]*K + start+k[None, :],
                         (m[:, None] < M) & (start+k[None, :] < K), other=0)
            acc3 = tl.dot(a3, b, acc3, out_dtype=tl.int32)
    result = acc.to(tl.float32) * tl.load(XS + m, m < M, other=0)[:, None]
    if COMPONENTS > 1:
        result += acc2.to(tl.float32) * tl.load(XS + M + m, m < M, other=0)[:, None]
    if COMPONENTS > 2:
        result += acc3.to(tl.float32) * tl.load(XS + 2*M + m, m < M, other=0)[:, None]
    if SPLIT == 1:
        result *= tl.load(WS + n, n < N, other=0)[None, :]
        if HAS_BIAS:
            result += tl.load(Bias+n, n < N, other=0)[None, :]
        if ACTIVATION == "silu":
            result = result / (1.0 + tl.exp(-result))
        tl.store(Y + m[:, None] * N + n[None, :], result, (m[:, None] < M) & (n[None, :] < N))
    else:
        tl.store(P + part * M * N + m[:, None] * N + n[None, :], result,
                 (m[:, None] < M) & (n[None, :] < N))


def integer_configuration(m, n, k):
    """Offline choices for the profiled short/medium batch-one encoder shapes."""
    base = (32, 64, 64, 4, 4, 3)
    if m > 512 or m < 5 or n < 1024 or k not in (1024, 4096):
        return base
    if k == 4096:
        if m <= 64:
            return (32, 64, 128, 8, 4, 2)
        return base if m <= 192 else (32, 64, 128, 4, 4, 2)
    if n >= 3072:
        return (32, 64, 128, 2 if m <= 64 else 1, 4, 2)
    if n >= 2048:
        return (32, 64, 128, 4 if m <= 64 else 2 if m <= 192 else 1, 4, 2)
    return (32, 64, 128, 8 if m <= 64 else 4, 4, 2)


def int8_matmul(x, mod, conv=False, components=1, activation="", norm=None, config=None, input_activation=""):
    from .kernels import _finish
    xx = x.transpose(1, 2) if conv else x
    if xx.ndim == 2:
        xx = xx.unsqueeze(0)
    b, t, k = xx.shape
    m, n = b*t, mod.out_features
    q = torch.empty((components, m, k), dtype=torch.int8, device=x.device)
    qs = torch.empty((components, m), dtype=torch.float32, device=x.device)
    y = torch.empty((b, t, n), dtype=torch.float32, device=x.device)
    bm, bn, bk, split, warps, stages = config or (integer_configuration(m, n, k) if components == 3 else (32,64,64,4,4,3))
    partial = torch.empty((split, m, n), device=x.device) if split > 1 else y
    _quantize[(m,)](xx, q, qs, norm.weight if norm is not None else None, norm.bias if norm is not None else None,
                    m, k, t, *xx.stride(), triton.next_power_of_2(k), components,
                    norm is not None, norm.eps if norm is not None else 0., INPUT_SILU=input_activation=="silu", enable_fp_fusion=False)
    _int8_dot[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
        q, qs, mod.packed_t, mod.scale, mod.bias, y, partial, m, n, k, mod.bias is not None,
        bm, bn, bk, split, components, ACTIVATION=activation, num_warps=warps, num_stages=stages)
    if split > 1:
        _finish[(triton.cdiv(m*n, 256),)](partial, mod.scale, mod.bias, y,
                                        m, n, t, t*n, n, 1, mod.bias is not None, split, 256, ACTIVATION=activation)
    return y.transpose(1, 2) if conv else y.reshape(*x.shape[:-1], n)


@triton.jit
def _attention(Q, K, V, U, BD, Mask, O,
               T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               QB: tl.constexpr, QH: tl.constexpr, QT: tl.constexpr,
               KB: tl.constexpr, KH: tl.constexpr, KT: tl.constexpr,
               VB: tl.constexpr, VH: tl.constexpr, VT: tl.constexpr,
               HAS_MASK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    m = tl.program_id(0)*BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    nn = tl.arange(0, BN)
    q = tl.load(Q + batch*QB + head*QH + m[:, None]*QT + d[None, :], m[:, None] < T, other=0)
    q += tl.load(U + head*D+d)[None, :]
    acc = tl.zeros((BM, D), tl.float32)
    maxv = tl.full((BM,), -1.0e30, tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    for start in range(tl.cdiv(T, BN)):
        n = start*BN + nn
        k = tl.load(K + batch*KB + head*KH + n[None, :]*KT + d[:, None], n[None, :] < T, other=0)
        scores = tl.dot(q, k, input_precision=PREC)
        pos = T-1-m[:, None]+n[None, :]
        bd = tl.load(BD + bh*T*(2*T-1) + m[:, None]*(2*T-1) + pos,
                     (m[:, None] < T) & (n[None, :] < T), other=0)
        scores = (scores + bd) * (D ** -0.5)
        valid = (m[:, None] < T) & (n[None, :] < T)
        if HAS_MASK:
            mask = tl.load(Mask + batch*T*T + m[:, None]*T + n[None, :], valid, other=1)
            valid = valid & ~mask
        scores = tl.where(valid, scores, -1.0e30)
        new_max = tl.maximum(maxv, tl.max(scores, 1))
        alpha = tl.exp(maxv-new_max)
        prob = tl.where(valid, tl.exp(scores-new_max[:, None]), 0.)
        denom = denom*alpha + tl.sum(prob, 1)
        acc *= alpha[:, None]
        v = tl.load(V + batch*VB + head*VH + n[:, None]*VT + d[None, :], n[:, None] < T, other=0)
        acc = tl.dot(prob, v, acc, input_precision=PREC)
        maxv = new_max
    result = acc / tl.where(denom > 0, denom, 1.)[:, None]
    tl.store(O + batch*T*H*D + m[:, None]*H*D + head*D + d[None, :], result, m[:, None] < T)


@triton.jit
def _position_scores(Q, P, V, Y, T:tl.constexpr, H:tl.constexpr, D:tl.constexpr,
                     QB:tl.constexpr,QH:tl.constexpr,QT:tl.constexpr,
                     PB:tl.constexpr,PH:tl.constexpr,PT:tl.constexpr,
                     BM:tl.constexpr,BN:tl.constexpr):
    bh=tl.program_id(2);batch=bh//H;head=bh%H
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN);d=tl.arange(0,D)
    q=tl.load(Q+batch*QB+head*QH+m[:,None]*QT+d[None,:],m[:,None]<T,other=0)
    q+=tl.load(V+head*D+d)[None,:]
    p=tl.load(P+batch*PB+head*PH+n[None,:]*PT+d[:,None],n[None,:]<2*T-1,other=0)
    y=tl.dot(q,p,input_precision='tf32x3')
    tl.store(Y+bh*T*(2*T-1)+m[:,None]*(2*T-1)+n[None,:],y,(m[:,None]<T)&(n[None,:]<2*T-1))


def relative_attention(att, query, key, value, mask, pos_emb, cache=None, *, position_dot=False):
    if cache is not None or query is not key or query is not value:
        raise ValueError("attention experiment supports offline self-attention only")
    q, k, v = att.forward_qkv(query, key, value)
    b, h, t, d = q.shape
    p = att.linear_pos(pos_emb).view(pos_emb.shape[0], -1, h, d).transpose(1, 2)
    # Short-sequence screen wins; larger relative-position matrices were tied
    # or slightly slower, so retain the existing matmul for those shapes.
    if position_dot and t <= 192:
        bd=torch.empty((b,h,t,2*t-1),device=q.device,dtype=q.dtype)
        _position_scores[(triton.cdiv(t,16),triton.cdiv(2*t-1,32),b*h)](
            q,p,att.pos_bias_v,bd,t,h,d,*q.stride()[:3],
            p.stride(0) if p.shape[0]>1 else 0,*p.stride()[1:3],16,32,num_warps=4)
    else:
        bd = torch.matmul(q + att.pos_bias_v[None, :, None, :], p.transpose(-2, -1))
    out = torch.empty((b, t, h*d), device=q.device, dtype=q.dtype)
    _attention[(triton.cdiv(t, 16), b*h)](
        q, k, v, att.pos_bias_u, bd, mask, out, t, h, d,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], mask is not None,
        16, 32, "tf32x3", num_warps=4, num_stages=1)
    return att.linear_out(out)
