"""Int8MmaLayer: 1-bit weights expanded to INT8 on chip, residual INT8x3 activations.

Two kernels per call (both on the current stream, graph capturable):
  1. b1_quantize: x (any strides) -> q int8 [3, M, ldq] + qs [3, M], the INT8x3 quantizer's arithmetic bitwise;
  2. b1_dot (fragment layout, kernels.py) or b1_fallback (packed export layout, for shapes whose padded fragment
     layout would exceed the 1.10x storage budget; never a Parakeet shape).
`project` is the fused op the projection provider calls: the INT8x3 quantizer (ternary.fused_kernels._quantize, with
LayerNorm / input SiLU fused) + b1_dot with an optional SiLU epilogue. `dot` is the lower-level entry on prequantized
activations and caller-owned outputs.
"""
from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from kea_kernels import _nvrtc as cuda

from kea_kernels.binary import configs, kernels
from kea_kernels.binary import bits as api


def _round_up(v: int, m: int) -> int:
    return -(-v // m) * m


class Int8MmaLayer:
    """One layer resident as 1-bit fragment words (or the raw packed codes) plus FP32 scale/bias."""

    def __init__(self, n: int, k: int, device: torch.device, words: Tensor | None, codes: Tensor | None,
                 scale: Tensor, bias: Tensor | None):
        self.out_features, self.in_features, self.device = n, k, device
        self.words, self.codes, self.scale, self.bias = words, codes, scale, bias

    @classmethod
    def prepare(cls, codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int, device) -> "Int8MmaLayer":
        n, k = api.validate_weights(codes, scale, bias, in_features)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Int8MmaLayer runs on CUDA devices only")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        with torch.no_grad():
            row_bytes = api.packed_row_bytes(k)
            budget = (n * row_bytes * 11) // 10
            if kernels.fragment_bytes(n, k) <= budget:
                words = kernels.relayout(api.unpack_bits(codes.to(device), k))
                raw = None
            else:
                words = None
                raw = codes.to(device=device, copy=True).contiguous()
            scale = scale.to(device=device, dtype=torch.float32, copy=True).contiguous()
            bias = None if bias is None else bias.to(device=device, dtype=torch.float32, copy=True).contiguous()
        return cls(n, k, device, words, raw, scale, bias)

    @property
    def fast(self) -> bool:
        return self.words is not None

    def weight_bytes(self) -> int:
        return api.resident_bytes(self.words, self.codes, self.scale, self.bias)

    # --- calls ------------------------------------------------------------------------------------
    def _check(self, x: Tensor, kind: str) -> None:
        if not isinstance(x, Tensor):
            raise ValueError("input must be a tensor")
        if x.dtype != torch.float32:
            raise ValueError(f"expected FP32 input, got {x.dtype}")
        if x.device != self.device:
            raise ValueError(f"input on {x.device}, layer on {self.device}")
        if kind == "linear" and (x.dim() < 1 or x.shape[-1] != self.in_features):
            raise ValueError(f"expected [..., {self.in_features}] input, got {tuple(x.shape)}")
        if kind == "conv" and (x.dim() != 3 or x.shape[1] != self.in_features):
            raise ValueError(f"expected [B, {self.in_features}, T] input, got {tuple(x.shape)}")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("Int8MmaLayer is inference only: input requires grad")

    def linear(self, x: Tensor) -> Tensor:
        self._check(x, "linear")
        lead = tuple(x.shape[:-1])
        m = math.prod(lead)
        if m == 0:
            return torch.empty(*lead, self.out_features, device=self.device, dtype=torch.float32)
        if x.dim() == 1:
            rows = (0, 0, x.stride(0), 1)
        elif x.dim() == 2:
            rows = (0, x.stride(0), x.stride(1), m)
        else:
            if x.dim() > 3:
                try:
                    x = x.view(-1, x.shape[-2], x.shape[-1])
                except RuntimeError:
                    x = x.reshape(-1, x.shape[-2], x.shape[-1])
            rows = (x.stride(0), x.stride(1), x.stride(2), x.shape[1])
        y = torch.empty(m, self.out_features, device=self.device, dtype=torch.float32)
        self._run(x, rows, m, y)
        return y.view(*lead, self.out_features)

    def conv(self, x: Tensor) -> Tensor:
        self._check(x, "conv")
        b, _, t = x.shape
        y = torch.empty(b, t, self.out_features, device=self.device, dtype=torch.float32)
        if b * t:
            self._run(x, (x.stride(0), x.stride(2), x.stride(1), t), b * t, y.view(b * t, self.out_features))
        return y.transpose(1, 2)

    def _run(self, x: Tensor, rows: tuple, m: int, y: Tensor) -> None:
        q, qs, ldq = quantize(x, rows, m, self.in_features)
        self.dot(q, qs, ldq, m, y)

    def dot(self, q: Tensor, qs: Tensor, ldq: int, m: int, y: Tensor, *, silu: bool = False,
            config: kernels.DotConfig | None = None, order: int = 0) -> None:
        """y[:m] (row-major, row stride y.stride(0)) from q int8 [3, m, ldq] and qs FP32 [3, m]."""
        dot(self, q, qs, ldq, m, y, silu=silu, config=config, order=order)

    def project(self, x: Tensor, *, conv: bool = False, components: int = 3, activation: str = "",
                norm=None, input_activation: str = "") -> Tensor:
        """The fused provider op (ternary.encoder4.ExpandedProjection.__call__ semantics).

        Activations are quantized by ternary.fused_kernels._quantize (fused LayerNorm and input SiLU), so q and qs are
        bitwise the ternary expanded path's; the matmul and epilogue are b1_dot."""
        if components != 3:
            raise ValueError("Int8MmaLayer implements INT8x3 arithmetic only (components=3)")
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
            if k % 16 == 0 and self.fast:
                q, qs = m1_quantize(xx, norm, input_activation)
                ldq = k
            else:  # synthetic ragged shapes only: unfused LayerNorm / SiLU, then b1_quantize
                z = xx
                if norm is not None:
                    z = torch.nn.functional.layer_norm(z, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
                if input_activation == "silu":
                    z = torch.nn.functional.silu(z)
                q, qs, ldq = quantize(z, (z.stride(0), z.stride(1), z.stride(2), t), m, k)
            self.dot(q, qs, ldq, m, y.view(m, n), silu=activation == "silu")
        if conv:
            return y.transpose(1, 2)
        return y if x.dim() == 3 else y.view(*x.shape[:-1], n)


def quantize(x: Tensor, rows: tuple, m: int, k: int) -> tuple[Tensor, Tensor, int]:
    """b1_quantize: (x, (XB, XT, XK, T)) -> q [3, m, ldq] (zero padded), qs [3, m], ldq."""
    xb, xt, xk, t = rows
    ldq = _round_up(k, 16)
    q = torch.empty(3, m, ldq, device=x.device, dtype=torch.int8)
    qs = torch.empty(3, m, device=x.device, dtype=torch.float32)
    source, name = kernels.quantize_source()
    cuda.load(source, name, 0).launch(
        ((m + 3) // 4, 1, 1), (128, 1, 1),
        [cuda.ptr(x), cuda.i64(xb), cuda.i64(xt), cuda.i64(xk), cuda.i32(max(t, 1)),
         cuda.ptr(q), cuda.ptr(qs), cuda.i32(m), cuda.i32(k), cuda.i32(ldq)])
    return q, qs, ldq


def m1_quantize(xx: Tensor, norm=None, input_activation: str = "") -> tuple[Tensor, Tensor]:
    """ternary.fused_kernels._quantize at three parts, called exactly as ExpandedProjection does."""
    import triton
    from kea_kernels.ternary.fused_kernels import _quantize
    b, t, k = xx.shape
    m = b * t
    q = torch.empty(3, m, k, device=xx.device, dtype=torch.int8)
    qs = torch.empty(3, m, device=xx.device, dtype=torch.float32)
    _quantize[(m,)](xx, q, qs, norm.weight if norm is not None else None, norm.bias if norm is not None else None,
                    m, k, t, *xx.stride(), triton.next_power_of_2(k), 3, norm is not None,
                    norm.eps if norm is not None else 0., INPUT_SILU=input_activation == "silu", enable_fp_fusion=False)
    return q, qs


def dot(layer: Int8MmaLayer, q: Tensor, qs: Tensor, ldq: int, m: int, y: Tensor, *, silu: bool = False,
            config: kernels.DotConfig | None = None, order: int = 0) -> None:
    """order selects the epilogue's first FP32 contraction (kernels.py docstring)."""
    n, k = layer.out_features, layer.in_features
    cstride, qs_stride = q.stride(0), qs.stride(0)
    if layer.fast:
        if ldq % 16 or q.stride(1) != ldq or q.stride(2) != 1 or q.data_ptr() % 16:
            raise ValueError("fast path needs 16-byte aligned rows of q")
        config = config or configs.choose(m, n, k)
        if config.xchg == 3 and 127 * k >= 2**20:  # packed lanes would overflow: exact INT32 exchange
            config = dataclasses.replace(config, xchg=1)
        source, name = kernels.dot_source(config, silu, order)
        ws_ints = config.workspace_ints(m, n)
        workspace = torch.empty(ws_ints, device=q.device, dtype=torch.int32) if ws_ints else None
        cuda.load(source, name, config.shared_bytes).launch(
            config.grid(m, n), (config.threads, 1, 1),
            [cuda.ptr(q), cuda.i64(cstride), cuda.i32(ldq), cuda.i32(_round_up(k, 16)), cuda.ptr(qs),
             cuda.i32(qs_stride), cuda.ptr(layer.words), cuda.i32(layer.words.shape[0]),
             cuda.i32(layer.words.shape[1]), cuda.ptr(layer.scale), cuda.ptr(layer.bias), cuda.ptr(y),
             cuda.i32(y.stride(0)), cuda.i32(m), cuda.i32(n), cuda.ptr(workspace)])
    else:
        source, name = kernels.fallback_source()
        total = m * n
        cuda.load(source, name, 0).launch(
            ((total + 127) // 128, 1, 1), (128, 1, 1),
            [cuda.ptr(q), cuda.i64(cstride), cuda.i32(ldq), cuda.ptr(qs), cuda.i32(qs_stride),
             cuda.ptr(layer.codes), cuda.i32(layer.codes.shape[1]), cuda.ptr(layer.scale), cuda.ptr(layer.bias),
             cuda.ptr(y), cuda.i32(y.stride(0)), cuda.i32(m), cuda.i32(n), cuda.i32(k), cuda.i32(int(silu) | 2 * order)])

