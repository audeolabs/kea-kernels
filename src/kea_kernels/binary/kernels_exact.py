"""Exact deferred split-K (XCHG 5) for the hybrid stack.

XCHG 4 would hand the row epilogue FP32 split partials per K slice, and a small residual part can vanish in a slice
whose partial is large and is then missing after the slices cancel. XCHG 5 keeps the K reduction in integers up to
the row epilogue: slice z = blockIdx.z packs its three INT32 part sums into one int64 (COMMON's pack3: 21-bit signed
lanes, linear, exact while 127 K < 2^20) and writes the low and high 32-bit words to two int32 planes, i.e. int32
[CLUSTER, 2, M, N]; no scale is read. row_exact._row_epilogue_exact rebuilds the int64 words, sums the slices in
int64, unpacks acc0..acc2 and applies the SPLIT == 1 epilogue before the residual, LayerNorm and INT8x3 quantization,
so the projection value is bitwise b1_dot's exact in-kernel finish.

Source: the dot source of the same tiles generated as XCHG 4 (rank = blockIdx.z, no cluster attribute), with
asserted edits: rank and template argument XCHG = 5, plus the XCHG 5 branch.
"""
from __future__ import annotations

import dataclasses

from kea_kernels.binary import kernels
from kea_kernels.binary.kernels import DotConfig

XCHG = 5
PACK_LIMIT = 2**20  # pack3 lanes are exact while 127 * K < 2^20

BRANCH = r'''    } else if constexpr (XCHG == 5) {
        // Exact deferred split-K (kernels_exact.py): slice z packs (acc0, acc1, acc2) into one int64 per
        // element and writes its low / high words to the int32 planes ((int *)Y) + (2 z) * M * ldy and
        // + (2 z + 1) * M * ldy, i.e. [CLUSTER, 2, M, N] with ldy == N. No scale is read here.
        if (wk == 0) {
            int *lo = reinterpret_cast<int *>(Y) + (long long)(2 * rank) * M * ldy;
            int *hi = lo + (long long)M * ldy;
            #pragma unroll
            for (int i = 0; i < RM; ++i)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int m = m0 + (wm * RM + i) * 16 + g + 8 * h;
                    if (m >= M) continue;
                    #pragma unroll
                    for (int j = 0; j < RN; ++j) {
                        const int n = n0 + (wn * RN + j) * 8 + s4 * 2;
                        const long long off = (long long)m * ldy + n;
                        const long long a = pack3(acc[0][i][j][2 * h], acc[1][i][j][2 * h], acc[2][i][j][2 * h]);
                        const long long b = pack3(acc[0][i][j][2 * h + 1], acc[1][i][j][2 * h + 1], acc[2][i][j][2 * h + 1]);
                        if (n + 1 < N && (ldy % 2 == 0)) {
                            *reinterpret_cast<int2 *>(lo + off) = make_int2((int)a, (int)b);
                            *reinterpret_cast<int2 *>(hi + off) = make_int2((int)(a >> 32), (int)(b >> 32));
                        } else {
                            if (n < N) { lo[off] = (int)a; hi[off] = (int)(a >> 32); }
                            if (n + 1 < N) { lo[off + 1] = (int)b; hi[off + 1] = (int)(b >> 32); }
                        }
                    }
                }
        }
    } else if constexpr (XCHG == 4) {
'''


@dataclasses.dataclass(frozen=True)
class ExactDeferredConfig(DotConfig):
    """A DotConfig with xchg = 5: CLUSTER K slices over blockIdx.z into int32 word planes [CLUSTER, 2, M, N]."""

    def __post_init__(self):
        if self.xchg != XCHG:
            raise ValueError("ExactDeferredConfig needs xchg = 5")

    @property
    def exact(self) -> bool:
        return True

    def workspace_ints(self, m: int, n: int) -> int:
        return 0  # the slice planes are the output buffer

    def label(self) -> str:
        return base(self, xchg=4).label().replace("_d", "_e", 1)  # "_e": exact packed integer slices

    def validate(self) -> None:
        base(self, xchg=4).validate()
        if self.cluster < 2:
            raise ValueError("exact deferred slices need cluster >= 2")
        if self.pf:
            raise ValueError("XCHG 5 reads no scales; pf must be 0")

    @classmethod
    def of(cls, config: DotConfig) -> "ExactDeferredConfig":
        return cls(*[getattr(config, f.name) if f.name not in ("xchg", "pf") else {"xchg": XCHG, "pf": 0}[f.name]
                     for f in dataclasses.fields(DotConfig)])


def base(config: DotConfig, **changes) -> DotConfig:
    """The plain frozen DotConfig of the same fields (subclass properties dropped)."""
    values = {f.name: getattr(config, f.name) for f in dataclasses.fields(DotConfig)}
    values.update(changes)
    return DotConfig(**values)


def _replace(source: str, old: str, new: str, count: int = 1) -> str:
    found = source.count(old)
    if found != count:
        raise AssertionError(f"exact-deferred transform expected {count} x {old[:70]!r}, found {found}")
    return source.replace(old, new)


def dot_source(config: DotConfig, order: int = 0) -> tuple[str, str]:
    """(source, kernel name) of the XCHG 5 kernel for `config`'s tiles."""
    cfg = ExactDeferredConfig.of(config)
    cfg.validate()
    frozen = base(cfg, xchg=4)
    source, name = kernels.dot_source(frozen, False, order)
    new_name = f"b1_dot_exact_{cfg.label()}_o{order}"
    source = _replace(source, f"\n{name}(", f"\n{new_name}(")
    source = _replace(source, "(XCHG == 4 ? (int)blockIdx.z : (int)cluster_rank())",
                      "(XCHG == 4 || XCHG == 5 ? (int)blockIdx.z : (int)cluster_rank())")
    source = _replace(source, "    } else if constexpr (XCHG == 4) {\n", BRANCH)
    args = f"b1_dot<{cfg.wm}, {cfg.wn}, {cfg.wk}, {cfg.rm}, {cfg.rn}, {cfg.stages}, {cfg.cluster}, {2 * order}, 4, 0>("
    source = _replace(source, args, args.replace(", 4, 0>(", f", {XCHG}, 0>("))
    return source, new_name


def shared_bytes(config: DotConfig) -> int:
    return base(ExactDeferredConfig.of(config), xchg=4).shared_bytes
