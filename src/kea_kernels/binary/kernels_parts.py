"""Activation part counts 1 / 2 / 3 for b1_dot.

The generated source text is byte-identical to the research runtime's (its comments use its names; see kernels.py).

The INT8x3 quantizer at COMPONENTS = P splits each activation row into P residual INT8 parts; _int8_dot combines them
with SPLIT == 1 as
    P = 1: r = acc0 * s0
    P = 2: r = acc0 * s0 + acc1 * s1     (LLVM contracts to fma(acc0, s0, acc1 * s1) or fma(acc1, s1, acc0 * s0))
    P = 3: r = fma(acc2, s2, <P = 2 form>)
then y = r * alpha (+ bias) and optionally SiLU, exactly as the x3 kernel does. This module produces that kernel for
any P from the x3 source (kernels.py) with asserted edits:
  * a PARTS constexpr; the A tile, the cp.async issue loop and the ldmatrix / IMMA loop cover only the P parts (A rows
    c * BMR .. for c < P), so x1 / x2 run one third / two thirds of the IMMAs;
  * QS loads of absent parts are guarded to 0 (qs is [P, M]);
  * the epilogue, fragment epilogue and split partial use comb<P> (the forms above; flags bit 1 picks the P = 2
    contraction as for x3), SiLU unchanged.
The accumulator arrays keep their three part slots: slots c >= P are never touched by an IMMA and stay 0, so the
exchange paths (WK reduction, cluster INT32 / packed int64 exchange, deferred slices) carry exact zeros for them.
Tile configurations are the x3 tables.
"""
from __future__ import annotations

import dataclasses

from kea_kernels.binary import kernels
from kea_kernels.binary.kernels import DotConfig

PART_FUNCTIONS = r'''
// Part-count-aware combinations (M1's _int8_dot with COMPONENTS = P, SPLIT == 1).
template <int P> DEVI float comb(int a0, int a1, int a2, float s0, float s1, float s2, int flags) {
    if constexpr (P == 1) {
        return __fmul_rn(__int2float_rn(a0), s0);
    } else {
        float r = (flags & 2) ? __fmaf_rn(__int2float_rn(a1), s1, __fmul_rn(__int2float_rn(a0), s0))
                              : __fmaf_rn(__int2float_rn(a0), s0, __fmul_rn(__int2float_rn(a1), s1));
        if constexpr (P == 3) r = __fmaf_rn(__int2float_rn(a2), s2, r);
        return r;
    }
}
template <int P> DEVI float epilogue_p(int a0, int a1, int a2, float s0, float s1, float s2, float alpha, const float *bias, int n, int flags) {
    float r = comb<P>(a0, a1, a2, s0, s1, s2, flags);
    r = bias ? __fmaf_rn(r, alpha, bias[n]) : __fmul_rn(r, alpha);
    if (flags & 1) r = div_full(r, __fadd_rn(ex2_approx(__fmul_rn(-r, 1.44269502162933349609375f)), 1.0f));
    return r;
}
template <int P> DEVI float epilogue_vp(int a0, int a1, int a2, float s0, float s1, float s2, float alpha, bool has_bias, float bias, int flags) {
    float r = comb<P>(a0, a1, a2, s0, s1, s2, flags);
    r = has_bias ? __fmaf_rn(r, alpha, bias) : __fmul_rn(r, alpha);
    if (flags & 1) r = div_full(r, __fadd_rn(ex2_approx(__fmul_rn(-r, 1.44269502162933349609375f)), 1.0f));
    return r;
}
template <int P> DEVI float partial_p(int a0, int a1, int a2, float s0, float s1, float s2) {
    return comb<P>(a0, a1, a2, s0, s1, s2, 0);  // M1's split partial (order 0 form)
}
'''


@dataclasses.dataclass(frozen=True)
class PartsConfig(DotConfig):
    """DotConfig plus the part count (shared memory follows the P-part A tile)."""
    parts: int = 3

    @property
    def shared_bytes(self) -> int:
        stage = (self.parts * self.bmr * 128 + (self.bn // 8) * 128) * self.wk
        reduce = 3 * self.bmr * (self.bn + 8) * 4 if self.cluster > 1 and self.xchg == 0 else 0
        groups = (self.wk - 1) * 3 * self.bmr * (self.bn + 8) * 4
        return max(self.stages * stage, reduce, groups)

    def label(self) -> str:
        return f"x{self.parts}_" + DotConfig.label(self)

    @classmethod
    def of(cls, config: DotConfig, parts: int) -> "PartsConfig":
        base = [getattr(config, f.name) for f in dataclasses.fields(DotConfig)]
        return cls(*base, parts=parts)


def _replace(source: str, old: str, new: str, count: int | None = 1) -> str:
    found = source.count(old)
    if (count is None and found == 0) or (count is not None and found != count):
        raise AssertionError(f"parts transform expected {count} x {old[:70]!r}, found {found}")
    return source.replace(old, new)


def transform(source: str, parts: int) -> str:
    """x3 dot source (COMMON + DOT + entry) -> P-part source."""
    if parts not in (1, 2, 3):
        raise ValueError("parts must be 1, 2 or 3")
    common, body = source[:len(kernels.COMMON)], source[len(kernels.COMMON):]
    if common != kernels.COMMON:
        raise AssertionError("unexpected source layout")
    common = _replace(common, "DEVI float finish(float sum, float alpha, const float *bias, int n, int flags) {",
                      PART_FUNCTIONS + "DEVI float finish(float sum, float alpha, const float *bias, int n, int flags) {")
    b = body
    b = _replace(b, "    constexpr int THREADS = WM * WN * WK * 32;\n",
                 "    constexpr int THREADS = WM * WN * WK * 32;\n"
                 f"    constexpr int PARTS = {parts};  // activation residual parts (plans/activation-precision.md)\n")
    b = _replace(b, "    constexpr int A_SUB = 3 * BMR * 128;", "    constexpr int A_SUB = PARTS * BMR * 128;")
    b = _replace(b, "for (int i = tid; i < 3 * BMR * 8 * WK; i += THREADS) {", "for (int i = tid; i < PARTS * BMR * 8 * WK; i += THREADS) {")
    b = _replace(b, "const int w = i / (3 * BMR * 8), rest = i - w * (3 * BMR * 8);",
                 "const int w = i / (PARTS * BMR * 8), rest = i - w * (PARTS * BMR * 8);")
    b = _replace(b, "            for (int c = 0; c < 3; ++c) {\n                #pragma unroll\n                for (int i = 0; i < RM; ++i) {\n"
                    "                    const int row = c * BMR + (wm * RM + i) * 16 + (lane & 15);",
                 "            for (int c = 0; c < PARTS; ++c) {\n                #pragma unroll\n                for (int i = 0; i < RM; ++i) {\n"
                 "                    const int row = c * BMR + (wm * RM + i) * 16 + (lane & 15);")
    b = _replace(b, "QS[2 * qs_stride + m]", "(PARTS > 2 ? QS[2 * qs_stride + m] : 0.f)", None)
    b = _replace(b, "QS[qs_stride + m]", "(PARTS > 1 ? QS[qs_stride + m] : 0.f)", None)
    b = _replace(b, "epilogue_v(", "epilogue_vp<PARTS>(", None)
    b = _replace(b, "epilogue(", "epilogue_p<PARTS>(", None)
    b = _replace(b, "partial(", "partial_p<PARTS>(", None)
    return common + b


def dot_source(config: DotConfig, silu: bool, order: int = 0, parts: int = 3) -> tuple[str, str]:
    """(source, kernel name) of the P-part b1_dot for a DotConfig."""
    base = DotConfig(*[getattr(config, f.name) for f in dataclasses.fields(DotConfig)])
    source, name = kernels.dot_source(base, silu, order)
    tag = f"x{parts}_"
    source = _replace(source, f"\n{name}(", f"\n{tag}{name}(")
    source = transform(source, parts)
    PartsConfig.of(config, parts).validate()
    return source, tag + name
