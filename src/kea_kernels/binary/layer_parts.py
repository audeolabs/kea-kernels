"""Int8MmaLayer with 1, 2 or 3 activation parts.

`Int8MmaPartsLayer` is Int8MmaLayer (layer.py, unchanged) with:
  * project(x, *, conv, components, activation, norm, input_activation) honouring components in {1, 2, 3} (PartCount
    overrides it on every provider call): the INT8 residual quantizer at COMPONENTS = components (fused LayerNorm /
    input SiLU), then the P-part b1_dot;
  * dot(q, qs, ldq, m, y, ..., parts=None): the P-part kernel, P = q.shape[0] unless given;
  * linear / conv at the variant's fixed part count PARTS: the first PARTS parts of b1_quantize (bitwise the 3-part
    quantizer, whose first P parts are its P-part output).
With P = 3 every call takes Int8MmaLayer's code path. Tile tables are the x3 ones. Shapes whose fragment layout
exceeds the storage budget use the 3-part fallback kernel: its missing parts are zero-padded, which gives the same
values (a zero part adds an exact 0).
"""
from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from kea_kernels import _nvrtc as cuda

from kea_kernels.binary import configs, kernels, kernels_parts
from kea_kernels.binary.layer import Int8MmaLayer, quantize


def _round_up(v: int, m: int) -> int:
    return -(-v // m) * m


def m1_quantize_parts(xx: Tensor, parts: int, norm=None, input_activation: str = "") -> tuple[Tensor, Tensor]:
    """The INT8 residual quantizer (ternary.fused_kernels._quantize) at COMPONENTS = parts, called as
    ExpandedProjection does."""
    import triton
    from kea_kernels.ternary.fused_kernels import _quantize
    b, t, k = xx.shape
    m = b * t
    q = torch.empty(parts, m, k, device=xx.device, dtype=torch.int8)
    qs = torch.empty(parts, m, device=xx.device, dtype=torch.float32)
    _quantize[(m,)](xx, q, qs, norm.weight if norm is not None else None, norm.bias if norm is not None else None,
                    m, k, t, *xx.stride(), triton.next_power_of_2(k), parts, norm is not None,
                    norm.eps if norm is not None else 0., INPUT_SILU=input_activation == "silu", enable_fp_fusion=False)
    return q, qs


class Int8MmaPartsLayer(Int8MmaLayer):
    PARTS = 3

    # --- matmul -----------------------------------------------------------------------------------
    def dot(self, q: Tensor, qs: Tensor, ldq: int, m: int, y: Tensor, *, silu: bool = False,
            config: kernels.DotConfig | None = None, order: int = 0, parts: int | None = None) -> None:
        parts = q.shape[0] if parts is None else parts
        if parts not in (1, 2, 3) or q.shape[0] != parts or qs.shape[0] != parts:
            raise ValueError(f"q / qs must hold {parts} parts")
        if not self.fast:  # 3-part fallback with zero parts appended (exact zeros)
            if parts < 3:
                q = torch.cat([q, q.new_zeros(3 - parts, *q.shape[1:])])
                qs = torch.cat([qs, qs.new_zeros(3 - parts, qs.shape[1])])
            return Int8MmaLayer.dot(self, q, qs, ldq, m, y, silu=silu, config=config, order=order)
        if parts == 3:
            return Int8MmaLayer.dot(self, q, qs, ldq, m, y, silu=silu, config=config, order=order)
        dot_parts(self, q, qs, ldq, m, y, silu=silu, config=config, order=order, parts=parts)

    # --- provider op ------------------------------------------------------------------------------
    def project(self, x: Tensor, *, conv: bool = False, components: int | None = None, activation: str = "",
                norm=None, input_activation: str = "") -> Tensor:
        parts = self.PARTS if components is None else components
        if parts not in (1, 2, 3):
            raise ValueError(f"components must be 1, 2 or 3, got {components!r}")
        if parts == 3:
            return Int8MmaLayer.project(self, x, conv=conv, components=3, activation=activation, norm=norm,
                                        input_activation=input_activation)
        if activation not in ("", "silu") or input_activation not in ("", "silu"):
            raise ValueError(f"unsupported activation {activation!r} / {input_activation!r}")
        xx = x.transpose(1, 2) if conv else x
        self._check(xx, "linear")
        if xx.dim() == 2:
            xx = xx.unsqueeze(0)
        if xx.dim() != 3:
            raise ValueError("project expects [B, T, K] (or [T, K]) input")
        b, t, k = xx.shape
        m, n = b * t, self.out_features
        y = torch.empty(b, t, n, device=self.device, dtype=torch.float32)
        if m:
            q, qs = m1_quantize_parts(xx, parts, norm, input_activation)
            q, ldq = _aligned(q, k)
            self.dot(q, qs, ldq, m, y.view(m, n), silu=activation == "silu", parts=parts)
        if conv:
            return y.transpose(1, 2)
        return y if x.dim() == 3 else y.view(*x.shape[:-1], n)

    # --- BinaryLayer calls at the fixed part count ----------------------------------------------------
    def _run(self, x: Tensor, rows: tuple, m: int, y: Tensor) -> None:
        """linear / conv: b1_quantize (layer.py) gives the three residual parts; the first PARTS of them are exactly
        the quantizer at COMPONENTS = PARTS (its loop is sequential)."""
        if self.PARTS == 3:
            return Int8MmaLayer._run(self, x, rows, m, y)
        q, qs, ldq = quantize(x, rows, m, self.in_features)
        self.dot(q[:self.PARTS], qs[:self.PARTS], ldq, m, y, parts=self.PARTS)


def _aligned(q: Tensor, k: int) -> tuple[Tensor, int]:
    """q [P, M, K] -> rows padded with zeros to a multiple of 16 bytes (cp.async alignment) if needed."""
    if k % 16 == 0 and q.data_ptr() % 16 == 0:
        return q, k
    ldq = _round_up(k, 16)
    padded = q.new_zeros(q.shape[0], q.shape[1], ldq)
    padded[:, :, :k] = q
    return padded, ldq


def dot_parts(layer: Int8MmaLayer, q: Tensor, qs: Tensor, ldq: int, m: int, y: Tensor, *, silu: bool = False,
              config: kernels.DotConfig | None = None, order: int = 0, parts: int = 3) -> None:
    """layer.py's dot() with the P-part kernel (identical argument list and launch geometry)."""
    n, k = layer.out_features, layer.in_features
    if ldq % 16 or q.stride(1) != ldq or q.stride(2) != 1 or q.data_ptr() % 16:
        raise ValueError("fast path needs 16-byte aligned rows of q")
    config = config or configs.choose(m, n, k)
    if config.xchg == 3 and 127 * k >= 2**20:
        config = dataclasses.replace(config, xchg=1)
    cfg = kernels_parts.PartsConfig.of(config, parts)
    source, name = kernels_parts.dot_source(cfg, silu, order, parts)
    ws_ints = cfg.workspace_ints(m, n)
    workspace = torch.empty(ws_ints, device=q.device, dtype=torch.int32) if ws_ints else None
    cuda.load(source, name, cfg.shared_bytes).launch(
        cfg.grid(m, n), (cfg.threads, 1, 1),
        [cuda.ptr(q), cuda.i64(q.stride(0)), cuda.i32(ldq), cuda.i32(_round_up(k, 16)), cuda.ptr(qs),
         cuda.i32(qs.stride(0)), cuda.ptr(layer.words), cuda.i32(layer.words.shape[0]),
         cuda.i32(layer.words.shape[1]), cuda.ptr(layer.scale), cuda.ptr(layer.bias), cuda.ptr(y),
         cuda.i32(y.stride(0)), cuda.i32(m), cuda.i32(n), cuda.ptr(workspace)])


def variant(parts: int):
    """The layer class whose linear / conv run at `parts` activation parts."""
    return type(f"Int8MmaX{parts}Layer", (Int8MmaPartsLayer,), {"PARTS": parts})


LAYERS = {parts: variant(parts) for parts in (1, 2, 3)}
