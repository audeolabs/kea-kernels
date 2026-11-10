"""The row epilogue with an exact integer split-K finish (x3).

`_row_epilogue_exact` is ternary.row_fusion._row_epilogue (same arguments, same arithmetic after the projection
value, same launch rule) except for how the projection value is formed: instead of summing FP32 split partials it
rebuilds kernels_exact's packed int64 words from their int32 low / high planes [SPLIT, 2, M, N] (three INT32 part sums
in 21-bit lanes), sums the slices in int64, unpacks acc0, acc1, acc2 and applies the SPLIT == 1 epilogue,
    order 0: v = fma(acc0, s0, acc1 * s1)    order 1: v = fma(acc1, s1, acc0 * s0)
    v = fma(acc2, s2, v);  v = fma(v, alpha, bias) or v * alpha (a scalar mul.rn in inline PTX: see _mul_rn)
op for op what b1_dot's in-kernel finish (kernels.COMMON `epilogue`) computes, so the projection value is bitwise the
SPLIT == 1 result for every slice count. The value is then parked in Y and reloaded with a volatile load into
`0 + value`, exactly the shape of the SPLIT == 1 `value += load(P)`, and the residual add, LayerNorm and INT8x3
quantization that follow are _row_epilogue's code verbatim (enable_fp_fusion=False). Why the round trip: the rest of
the kernel must compile as _row_epilogue does, not just be written the same. ptxas contracts LLVM's f32x2 mul.rn +
add.rn pairs into FFMA despite the explicit rounding, and which pairs LLVM vectorizes depends on how the value was
produced; computed in registers, the LayerNorm / quantization differed by ulps. A CTA barrier separates the store and
the load because Triton may give them different layouts. Every load is 4-byte and contiguous, so Triton assigns the
same blocked layout as _row_epilogue.

Routing: `install()` (run on import) replaces ternary.row_fusion.finish_residual, which the row_fusions layer forwards
look up at call time, by a dispatcher: an ExactDeferredProjection goes to `finish_residual_exact`, anything else to
the original unchanged. ExactDeferredProjection is a DeferredProjection whose `partial` attribute raises, so the
original finish_residual can never read the integer slices as FP32 partials.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from kea_kernels.ternary.row_fusion import DeferredProjection


class ExactDeferredProjection(DeferredProjection):
    """slices: int32 [split, 2, M, N] low / high words of the packed INT32x3 K-slice sums; xs: the part scales
    [3, M]; order: the epilogue contraction form. `partial` (FP32 slices) deliberately raises."""

    def __init__(self, slices, scale, bias, split, shape, xs, order=0):  # noqa: D107 - no dataclass init
        self.slices, self.scale, self.bias, self.split, self.shape, self.xs, self.order = \
            slices, scale, bias, split, shape, xs, order

    @property
    def partial(self):
        raise RuntimeError("ExactDeferredProjection holds integer slices, not FP32 partials: finish it with "
                           "kea_kernels.binary.row_exact (row_fusion.finish_residual is patched by install())")

    def __repr__(self) -> str:
        return f"ExactDeferredProjection(split={self.split}, shape={self.shape}, order={self.order})"

    __eq__ = object.__eq__
    __hash__ = object.__hash__


@triton.jit
def _mul_rn(a, b):
    # One scalar mul.rn.f32 per element in inline PTX. LLVM's SLP vectorizer otherwise packs value * alpha
    # into mul.rn.f32x2, and ptxas contracted that f32x2 mul.rn with the following residual add.rn into an
    # FFMA (measured: 4 extra FFMA in SASS, x = fma(v, alpha, R) instead of R + round(v * alpha)). A
    # scalar mul.rn.f32 is kept as written (PTX ISA: explicit rounding is not contracted), as in _row_epilogue.
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _row_epilogue_exact(P, XS, XS_STRIDE, S, B, R, Y, Q, QS, NW, NB, OW, OB,
                        M: tl.constexpr, N: tl.constexpr, SPLIT: tl.constexpr, FACTOR: tl.constexpr,
                        HAS_BIAS: tl.constexpr, QUANTIZE: tl.constexpr, OUT_NORM: tl.constexpr, EPS: tl.constexpr,
                        O_EPS: tl.constexpr, BLOCK: tl.constexpr, ORDER: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, BLOCK)
    # Exact split-K: rebuild each slice's packed int64 from its word planes ((hi << 32) | zext(lo): a
    # register pair), sum in int64, then unpack the three INT32 part sums (21-bit signed lanes) with
    # 32-bit sign extensions where the lane fits a word (fewer 64-bit ops; measured 0.1-0.4 us faster
    # per call than 64-bit shifts and where() at M = 38-256).
    packed = tl.full((BLOCK,), 0, tl.int64)
    for part in tl.static_range(SPLIT):
        lo = tl.load(P + (2 * part) * M * N + row * N + n, n < N, other=0)
        hi = tl.load(P + (2 * part + 1) * M * N + row * N + n, n < N, other=0)
        packed += (hi.to(tl.int64) << 32) | lo.to(tl.uint32, bitcast=True).to(tl.int64)
    a0 = (packed.to(tl.int32) << 11) >> 11
    rest = (packed - a0.to(tl.int64)) >> 21
    a1 = (rest.to(tl.int32) << 11) >> 11
    a2 = ((rest - a1.to(tl.int64)) >> 21).to(tl.int32)
    f0 = a0.to(tl.float32)
    f1 = a1.to(tl.float32)
    f2 = a2.to(tl.float32)
    s0 = tl.load(XS + row)
    s1 = tl.load(XS + XS_STRIDE + row)
    s2 = tl.load(XS + 2 * XS_STRIDE + row)
    if ORDER == 0:
        value = tl.fma(f0, s0, f1 * s1)
    else:
        value = tl.fma(f1, s1, f0 * s0)
    value = tl.fma(f2, s2, value)
    if HAS_BIAS:
        value = tl.fma(value, tl.load(S + n, n < N, other=0), tl.load(B + n, n < N, other=0))
    else:
        value = _mul_rn(value, tl.load(S + n, n < N, other=0))
    # Park the exact value in Y (overwritten below) and reload it with a volatile load (no store-to-load
    # forwarding), so that from here on the kernel is the SPLIT == 1 _row_epilogue token for token and
    # compiles the same way (module docstring). Triton may give the store and the load different layouts
    # (it does at 4 warps), i.e. threads may read elements other threads wrote: the CTA barrier orders
    # and makes visible those global writes.
    tl.store(Y + row * N + n, value, n < N)
    tl.debug_barrier()
    value = tl.full((BLOCK,), 0, tl.float32)
    value += tl.load(Y + row * N + n, n < N, other=0, volatile=True)
    # From here on: inference.row_fusion._row_epilogue verbatim (SPLIT == 1).
    x = tl.load(R + row * N + n, n < N, other=0) + value * FACTOR
    if OUT_NORM:
        mean = tl.sum(x, 0) / N
        d = tl.where(n < N, x - mean, 0.)
        x = d * tl.rsqrt(tl.sum(d * d, 0) / N + O_EPS) * tl.load(OW + n, n < N, other=0) + tl.load(OB + n, n < N, other=0)
    tl.store(Y + row * N + n, x, n < N)
    if QUANTIZE:
        mean = tl.sum(x, 0) / N
        d = tl.where(n < N, x - mean, 0.)
        x = d * tl.rsqrt(tl.sum(d * d, 0) / N + EPS) * tl.load(NW + n, n < N, other=0) + tl.load(NB + n, n < N, other=0)
        for part in tl.static_range(3):
            scale = tl.maximum(tl.max(tl.abs(x), 0) / 127., 1.e-30)
            q = tl.extra.cuda.libdevice.nearbyint(x / scale).to(tl.int8)
            tl.store(Q + part * M * N + row * N + n, q, n < N)
            tl.store(QS + part * M + row, scale)
            x = x - q.to(tl.float32) * scale


def finish_residual_exact(provider, value: ExactDeferredProjection, residual, factor, norm=None, out_norm=None):
    """inference.row_fusion.finish_residual for ExactDeferredProjection (same checks, outputs, cache)."""
    if not isinstance(value, ExactDeferredProjection):
        raise TypeError("finish_residual_exact needs an ExactDeferredProjection")
    if not residual.is_contiguous() or residual.shape != value.shape:
        raise ValueError("expected contiguous B,T,C residual")
    n = residual.shape[-1]
    m = residual.numel() // n
    if value.slices.dtype != torch.int32 or tuple(value.slices.shape) != (value.split, 2, m, n) \
            or not value.slices.is_contiguous():
        raise ValueError("expected contiguous int32 slice planes [split, 2, M, N]")
    if value.xs.dtype != torch.float32 or value.xs.shape[0] != 3 or value.xs.shape[1] != m or value.xs.stride(1) != 1:
        raise ValueError("expected FP32 part scales [3, M]")
    y = torch.empty_like(residual)
    q = torch.empty(3, m, n, device=y.device, dtype=torch.int8) if norm is not None else provider.null
    qs = torch.empty(3, m, device=y.device) if norm is not None else provider.null
    warps = getattr(provider, "row_warps", 4)
    if warps == "adaptive":
        warps = 8 if m <= 192 else 4
    _row_epilogue_exact[(m,)](value.slices, value.xs, value.xs.stride(0), value.scale, value.bias, residual, y, q, qs,
                              norm.weight if norm is not None else None, norm.bias if norm is not None else None,
                              out_norm.weight if out_norm is not None else None,
                              out_norm.bias if out_norm is not None else None,
                              m, n, value.split, factor, value.bias is not None, norm is not None, out_norm is not None,
                              norm.eps if norm is not None else 0., out_norm.eps if out_norm is not None else 0.,
                              triton.next_power_of_2(n), value.order, num_warps=warps, enable_fp_fusion=False)
    if norm is not None:
        key = (y.data_ptr(), m, n, id(norm))
        if key in provider.prequantized:
            raise RuntimeError("unconsumed fused activation")
        provider.prequantized[key] = (q, qs)
    return y


_ORIGINAL = None


def install() -> None:
    """Idempotent: route row_fusion.finish_residual (module docstring)."""
    global _ORIGINAL
    from kea_kernels.ternary import row_fusion
    if getattr(row_fusion.finish_residual, "exact_dispatch", False):
        return
    _ORIGINAL = original = row_fusion.finish_residual

    def finish_residual(provider, value, residual, factor, norm=None, out_norm=None):
        if isinstance(value, ExactDeferredProjection):
            return finish_residual_exact(provider, value, residual, factor, norm, out_norm)
        return original(provider, value, residual, factor, norm, out_norm)

    finish_residual.exact_dispatch = True
    finish_residual.original = original
    row_fusion.finish_residual = finish_residual


install()
