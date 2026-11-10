"""The hybrid provider at two activation parts (Kea-B's x2 default).

Stack: `PartCount(HybridProjectionX2(table, names), 2)` (PartCount forwards attribute writes to the provider) inside
`exact_hybrid_pipeline_x2(model, provider)`, i.e. the hybrid context stack with row_exact_x2's router installed. The
table's layers are layer_parts' 2-part variant.

HybridProjectionX2 is HybridProjectionExact (hybrid_exact.py) with these 3-to-2 changes, and nothing else:
  * it serves components == 2 only (PartCount sets it on every call) and raises on anything else;
  * activations are the INT8 residual quantizer at COMPONENTS = 2 (layer_parts.m1_quantize_parts), or the cached
    row-epilogue quantization's parts 0 and 1 (q[:2], qs[:2]);
  * the in-kernel route is the layer's P-part b1_dot (kernels_parts) with the x3 tiles (configs.choose), finished by
    _row_epilogue with SPLIT = 1 when deferred;
  * the XCHG 5 route (hybrid_exact.choose_exact_deferred, the x3 table) is kernels_exact_x2's 2-part kernel, returned
    as an ExactDeferredProjectionX2 that row_exact_x2 finishes exactly;
  * fused members called alone use the layer's project(components=2).
Deferral decisions depend on (M, N, K), the weights and the defer flag only, so the route of every call equals the
x3 stack's.
"""
from __future__ import annotations

import contextlib

import torch

from kea_kernels import _nvrtc as cuda

from kea_kernels.binary import configs, kernels_exact, kernels_exact_x2, layer_parts, row_exact_x2
from kea_kernels.binary.hybrid import hybrid_pipeline
from kea_kernels.binary.hybrid_exact import HybridProjectionExact, choose_exact_deferred
from kea_kernels.binary.kernels import DotConfig

PARTS = 2


def is_x2_layer(layer) -> bool:
    return isinstance(layer, layer_parts.Int8MmaPartsLayer) and layer.PARTS == PARTS


def quantize_x2(xx, norm=None, input_activation: str = ""):
    """The INT8 residual quantizer at COMPONENTS = 2 (exactly layer_parts' project quantization)."""
    return layer_parts.m1_quantize_parts(xx, PARTS, norm, input_activation)


def dot_slices_x2(layer, q: torch.Tensor, qs: torch.Tensor, ldq: int, m: int, slices: torch.Tensor,
                  config: DotConfig, order: int = 0) -> None:
    """slices int32 [C, 2, m, N] <- packed P-part K-slice sums (P = q.shape[0] in (2, 3)) of b1_dot XCHG 5."""
    p = q.shape[0]
    n, k = layer.out_features, layer.in_features
    cfg = kernels_exact.ExactDeferredConfig.of(config)
    if p not in kernels_exact_x2.PARTS or qs.shape[0] != p:
        raise ValueError(f"q / qs must hold 2 or 3 parts, got {tuple(q.shape)} / {tuple(qs.shape)}")
    if not layer.fast:
        raise ValueError("exact deferred slices need the fragment layout (layer.fast)")
    if 127 * k >= kernels_exact.PACK_LIMIT:
        raise ValueError("packed int64 lanes would overflow for this K")
    if ldq % 16 or q.stride(1) != ldq or q.stride(2) != 1 or q.data_ptr() % 16:
        raise ValueError("fast path needs q [P, M, ldq] with 16-byte aligned rows")
    if qs.stride(1) != 1:
        raise ValueError("qs rows must be contiguous")
    if slices.dtype != torch.int32 or tuple(slices.shape) != (cfg.cluster, 2, m, n) or not slices.is_contiguous():
        raise ValueError(f"slices must be contiguous int32 [{cfg.cluster}, 2, {m}, {n}]")
    source, name = kernels_exact_x2.dot_source(cfg, order, p)
    cuda.load(source, name, kernels_exact_x2.shared_bytes(cfg, p)).launch(
        cfg.grid(m, n), (cfg.threads, 1, 1),
        [cuda.ptr(q), cuda.i64(q.stride(0)), cuda.i32(ldq), cuda.i32(-(-k // 16) * 16), cuda.ptr(qs),
         cuda.i32(qs.stride(0)), cuda.ptr(layer.words), cuda.i32(layer.words.shape[0]),
         cuda.i32(layer.words.shape[1]), cuda.ptr(layer.scale), cuda.ptr(layer.bias), cuda.ptr(slices),
         cuda.i32(n), cuda.i32(m), cuda.i32(n), cuda.ptr(None)])


class HybridProjectionX2(HybridProjectionExact):
    """HybridProjectionExact at two activation parts (module docstring). Use inside PartCount(·, 2)."""

    PARTS = PARTS

    def __init__(self, table, names=None):
        bad = [e.name for e in table.entries if not is_x2_layer(e.layer)]
        if bad:
            raise ValueError(f"HybridProjectionX2 needs 2-part layers (layer_parts, PARTS 2); not: {bad[:3]}")
        super().__init__(table, names)
        row_exact_x2.install()

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        from kea_kernels.ternary.row_fusion import DeferredProjection
        if components != PARTS:
            raise ValueError(f"HybridProjectionX2 serves INT8x2 calls only (components={components!r}); "
                             "wrap it in kea_kernels.binary.provider.PartCount(provider, 2)")
        if config is not None:
            raise ValueError("HybridProjectionX2 serves calls without tile configs")
        entry, offset, count = self._resolver._lookup(mod)
        layer = entry.layer
        if count != entry.out_features:  # a fused member called alone (never inside the pipeline)
            y = layer.project(x, conv=conv, components=PARTS, activation=activation, norm=norm,
                              input_activation=input_activation)
            return y[:, offset:offset + count] if conv else y[..., offset:offset + count]
        xx = x.transpose(1, 2) if conv else x
        if xx.dim() == 2:
            xx = xx.unsqueeze(0)
        b, t, k = xx.shape
        m, n = b * t, layer.out_features
        cached = self.prequantized.pop((xx.data_ptr(), m, k, id(norm)), None) if norm is not None else None
        if cached is not None:  # the row epilogue quantized 3 parts; parts 0, 1 are its 2-part output
            q, qs = cached
            if q.shape[0] < PARTS or qs.shape[0] < PARTS:
                raise ValueError(f"cached activation has {q.shape[0]} parts")
            q, qs = q[:PARTS], qs[:PARTS]
            self.calls["prequantized"] += 1
        else:
            q, qs = quantize_x2(xx, norm, input_activation)
            self.calls["quantized"] += 1
        deferred = (self.defer_epilogue and n == 1024 and mod.packed_t.data_ptr() not in self.no_defer_weights
                    and activation == "")
        if deferred:
            self.calls["deferred"] += 1
            cfg = choose_exact_deferred(m, n, k) if self.split_deferred and layer.fast and m > 0 else None
            if cfg is not None:
                slices = torch.empty(cfg.cluster, 2, m, n, device=xx.device, dtype=torch.int32)
                dot_slices_x2(layer, q, qs, k, m, slices, cfg)
                self.calls["exact_slices"] = self.calls.get("exact_slices", 0) + 1
                return row_exact_x2.ExactDeferredProjectionX2(slices, layer.scale, layer.bias, cfg.cluster, (b, t, n),
                                                              qs, 0, PARTS)
        y = torch.empty(b, t, n, device=xx.device, dtype=torch.float32)
        layer.dot(q, qs, k, m, y.view(m, n), silu=activation == "silu", config=configs.choose(m, n, k), parts=PARTS)
        if deferred:  # exact in-kernel finish; _row_epilogue with SPLIT = 1 adds residual, norm, quantization
            return DeferredProjection(y, layer.scale, layer.bias, 1, (b, t, n))
        return y.transpose(1, 2) if conv else y.reshape(*x.shape[:-1], n)


def check_write_forwarding(provider, inner) -> None:
    """A write to the wrapper must reach the inner provider (the hybrid contexts set defer_epilogue and
    no_defer_weights on the object they are given). Raises if it does not."""
    if provider is inner:
        return
    before = inner.defer_epilogue
    provider.defer_epilogue = not before
    reached = inner.defer_epilogue is (not before)
    provider.defer_epilogue = before
    if not reached or inner.defer_epilogue is not before:
        raise RuntimeError(f"{type(provider).__name__} does not forward attribute writes to the provider; "
                           "the hybrid stack would never defer")


def unwrap(provider):
    """The innermost provider of a chain of .provider / .inner wrappers."""
    seen = set()
    while id(provider) not in seen:
        seen.add(id(provider))
        nxt = provider.__dict__.get("provider") or provider.__dict__.get("inner")
        if nxt is None:
            break
        provider = nxt
    return provider


@contextlib.contextmanager
def exact_hybrid_pipeline_x2(model, provider):
    """hybrid.hybrid_pipeline with row_exact_x2's router; `provider` is PartCount(HybridProjectionX2(...), 2)."""
    inner = unwrap(provider)
    if not isinstance(inner, HybridProjectionX2):
        raise TypeError(f"expected a HybridProjectionX2 inside the adapter, got {type(inner).__name__}")
    check_write_forwarding(provider, inner)
    row_exact_x2.install()
    with hybrid_pipeline(model, provider):
        yield provider
