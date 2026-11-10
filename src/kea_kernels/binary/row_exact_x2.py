"""The row epilogue with an exact integer split-K finish at 2 activation parts.

`_row_epilogue_exact_parts` is row_exact._row_epilogue_exact with a PARTS constexpr in the one place where the part
count enters, the projection value:
    PARTS 2, order 0: v = fma(acc0, s0, acc1 * s1)     order 1: v = fma(acc1, s1, acc0 * s0)
    PARTS 3:          v = fma(acc2, s2, <PARTS 2 form>)                     (row_exact's kernel)
    v = fma(v, alpha, bias) or v * alpha (scalar inline-PTX mul.rn, as row_exact._mul_rn)
which is op for op the in-kernel finish of the P-part b1_dot (kernels_parts comb<P>). At PARTS 2 only scale rows 0
and 1 are read and the packed lane 2 (always 0, kernels_exact_x2) is not decoded. Everything after the value is
row_exact's kernel token for token, including its 3-part quantization of the next layer's input (the x2 consumer,
hybrid_x2.HybridProjectionX2, reads parts 0 and 1, which are the 2-part quantization because the quantizer loop is
sequential).

Routing: `install()` runs row_exact.install() first and then wraps the current ternary.row_fusion.finish_residual,
i.e. row_exact's router. ExactDeferredProjectionX2 goes to `finish_residual_exact_x2`; every other value goes to
row_exact's router unchanged. ExactDeferredProjectionX2 is a DeferredProjection whose `partial` raises and which is
NOT a row_exact.ExactDeferredProjection.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from kea_kernels.ternary.row_fusion import DeferredProjection

from kea_kernels.binary import row_exact


class ExactDeferredProjectionX2(DeferredProjection):
    """slices: int32 [split, 2, M, N] low / high words of the packed K-slice sums (lane 2 is 0);
    xs: the projection input's part scales [parts, M]; order: the epilogue contraction form."""

    def __init__(self, slices, scale, bias, split, shape, xs, order=0, parts=2):  # noqa: D107 - no dataclass init
        self.slices, self.scale, self.bias, self.split, self.shape = slices, scale, bias, split, shape
        self.xs, self.order, self.parts = xs, order, parts

    @property
    def partial(self):
        raise RuntimeError("ExactDeferredProjectionX2 holds integer slices, not FP32 partials: finish it with "
                           "kea_kernels.binary.row_exact_x2 (installed by its install())")

    def __repr__(self) -> str:
        return f"ExactDeferredProjectionX2(parts={self.parts}, split={self.split}, shape={self.shape}, order={self.order})"

    __eq__ = object.__eq__
    __hash__ = object.__hash__


@triton.jit
def _mul_rn(a, b):
    # row_exact._mul_rn (see there): a scalar mul.rn.f32 that ptxas does not contract with the residual add.
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _row_epilogue_exact_parts(P, XS, XS_STRIDE, S, B, R, Y, Q, QS, NW, NB, OW, OB,
                              M: tl.constexpr, N: tl.constexpr, SPLIT: tl.constexpr, FACTOR: tl.constexpr,
                              HAS_BIAS: tl.constexpr, QUANTIZE: tl.constexpr, OUT_NORM: tl.constexpr,
                              EPS: tl.constexpr, O_EPS: tl.constexpr, BLOCK: tl.constexpr, ORDER: tl.constexpr,
                              PARTS: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, BLOCK)
    # Exact split-K (row_exact): int64 sum of the packed slices, then the INT32 part sums of the PARTS lanes.
    packed = tl.full((BLOCK,), 0, tl.int64)
    for part in tl.static_range(SPLIT):
        lo = tl.load(P + (2 * part) * M * N + row * N + n, n < N, other=0)
        hi = tl.load(P + (2 * part + 1) * M * N + row * N + n, n < N, other=0)
        packed += (hi.to(tl.int64) << 32) | lo.to(tl.uint32, bitcast=True).to(tl.int64)
    a0 = (packed.to(tl.int32) << 11) >> 11
    rest = (packed - a0.to(tl.int64)) >> 21
    a1 = (rest.to(tl.int32) << 11) >> 11
    f0 = a0.to(tl.float32)
    f1 = a1.to(tl.float32)
    s0 = tl.load(XS + row)
    s1 = tl.load(XS + XS_STRIDE + row)
    if ORDER == 0:
        value = tl.fma(f0, s0, f1 * s1)
    else:
        value = tl.fma(f1, s1, f0 * s0)
    if PARTS == 3:
        a2 = ((rest - a1.to(tl.int64)) >> 21).to(tl.int32)
        value = tl.fma(a2.to(tl.float32), tl.load(XS + 2 * XS_STRIDE + row), value)
    if HAS_BIAS:
        value = tl.fma(value, tl.load(S + n, n < N, other=0), tl.load(B + n, n < N, other=0))
    else:
        value = _mul_rn(value, tl.load(S + n, n < N, other=0))
    # row_exact: park the exact value in Y, CTA barrier, volatile reload, so that from here on the kernel
    # is the SPLIT == 1 _row_epilogue token for token and compiles the same way.
    tl.store(Y + row * N + n, value, n < N)
    tl.debug_barrier()
    value = tl.full((BLOCK,), 0, tl.float32)
    value += tl.load(Y + row * N + n, n < N, other=0, volatile=True)
    # From here on: inference.row_fusion._row_epilogue verbatim (SPLIT == 1), 3-part quantization included.
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


def finish_residual_exact_x2(provider, value: ExactDeferredProjectionX2, residual, factor, norm=None, out_norm=None):
    """inference.row_fusion.finish_residual for ExactDeferredProjectionX2 (same checks, outputs and
    prequantized cache entry as _row_epilogue's: a 3-part q [3, M, N] / qs [3, M])."""
    if not isinstance(value, ExactDeferredProjectionX2):
        raise TypeError("finish_residual_exact_x2 needs an ExactDeferredProjectionX2")
    if value.parts not in (2, 3):
        raise ValueError(f"parts must be 2 or 3, got {value.parts!r}")
    if not residual.is_contiguous() or residual.shape != value.shape:
        raise ValueError("expected contiguous B,T,C residual")
    n = residual.shape[-1]
    m = residual.numel() // n
    if value.slices.dtype != torch.int32 or tuple(value.slices.shape) != (value.split, 2, m, n) \
            or not value.slices.is_contiguous():
        raise ValueError("expected contiguous int32 slice planes [split, 2, M, N]")
    xs = value.xs
    if xs.dtype != torch.float32 or xs.shape[0] != value.parts or xs.shape[1] != m or xs.stride(1) != 1:
        raise ValueError(f"expected FP32 part scales [{value.parts}, M] with unit row stride")
    y = torch.empty_like(residual)
    q = torch.empty(3, m, n, device=y.device, dtype=torch.int8) if norm is not None else provider.null
    qs = torch.empty(3, m, device=y.device) if norm is not None else provider.null
    warps = getattr(provider, "row_warps", 4)
    if warps == "adaptive":
        warps = 8 if m <= 192 else 4
    _row_epilogue_exact_parts[(m,)](value.slices, xs, xs.stride(0), value.scale, value.bias, residual, y, q, qs,
                                    norm.weight if norm is not None else None, norm.bias if norm is not None else None,
                                    out_norm.weight if out_norm is not None else None,
                                    out_norm.bias if out_norm is not None else None,
                                    m, n, value.split, factor, value.bias is not None, norm is not None,
                                    out_norm is not None, norm.eps if norm is not None else 0.,
                                    out_norm.eps if out_norm is not None else 0., triton.next_power_of_2(n),
                                    value.order, value.parts, num_warps=warps, enable_fp_fusion=False)
    if norm is not None:
        key = (y.data_ptr(), m, n, id(norm))
        if key in provider.prequantized:
            raise RuntimeError("unconsumed fused activation")
        provider.prequantized[key] = (q, qs)
    return y


def install() -> None:
    """Idempotent: row_exact's router, then this router on top of it (module docstring)."""
    from kea_kernels.ternary import row_fusion
    row_exact.install()
    previous = row_fusion.finish_residual
    if getattr(previous, "exact_x2_dispatch", False):
        return
    if not getattr(previous, "exact_dispatch", False):
        raise RuntimeError("row_exact's router is not installed under the x2 router")

    def finish_residual(provider, value, residual, factor, norm=None, out_norm=None):
        if isinstance(value, ExactDeferredProjectionX2):
            return finish_residual_exact_x2(provider, value, residual, factor, norm, out_norm)
        return previous(provider, value, residual, factor, norm, out_norm)

    finish_residual.exact_dispatch = True      # row_exact.install() stays a no-op above this router
    finish_residual.exact_x2_dispatch = True
    row_fusion.finish_residual = finish_residual


def installed() -> bool:
    from kea_kernels.ternary import row_fusion
    return bool(getattr(row_fusion.finish_residual, "exact_x2_dispatch", False))
