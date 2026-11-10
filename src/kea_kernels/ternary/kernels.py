"""Tile-local decoding of Parakeet v1 codes: 00=0, 01=+1, 10=-1.

Only [ceil(K/16), N] packed int32 words are resident. No expanded weight matrix,
activation quantization buffer, host synchronization, or runtime autotuning.
FP32 inputs use three BF16 components in registers (or optional tf32x3);
scales, accumulation, and outputs remain FP32. No single-BF16 rounding of
activations. These are high-accuracy approximations, not bitwise FP32 BLAS.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(X, W, S, Bias, Y, Partial, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
          X_B: tl.constexpr, X_T: tl.constexpr, X_K: tl.constexpr,
          Y_B: tl.constexpr, Y_T: tl.constexpr, Y_N: tl.constexpr,
          HAS_BIAS: tl.constexpr, BN: tl.constexpr, RK: tl.constexpr, SPLIT: tl.constexpr,
          ACTIVATION: tl.constexpr = ""):
    m = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    kw = part * (RK // 16) + tl.arange(0, RK // 16)
    lane = tl.arange(0, 16)
    w = tl.load(W + kw[:, None] * N + n[None, :],
                (kw[:, None] < tl.cdiv(K, 16)) & (n[None, :] < N), other=0)
    c = ((w[:, None, :].to(tl.uint32) >> (lane[None, :, None] * 2)) & 3).reshape(RK, BN)
    k = part * RK + tl.arange(0, RK)
    x = tl.load(X + (m // T) * X_B + (m % T) * X_T + k * X_K, k < K, other=0).to(tl.float32)
    value = tl.sum(x[:, None] * ((c & 1).to(tl.float32) - (c >> 1).to(tl.float32)), axis=0)
    if SPLIT == 1:
        value *= tl.load(S + n, n < N, other=0)
        if HAS_BIAS:
            value += tl.load(Bias + n, n < N, other=0)
        if ACTIVATION == "silu":
            value = value / (1.0 + tl.exp(-value))
        tl.store(Y + (m // T) * Y_B + (m % T) * Y_T + n * Y_N, value, n < N)
    else:
        tl.store(Partial + part * M * N + m * N + n, value, n < N)


@triton.jit
def _matmul(X, W, S, Bias, Y, Partial,
            M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
            X_B: tl.constexpr, X_T: tl.constexpr, X_K: tl.constexpr,
            Y_B: tl.constexpr, Y_T: tl.constexpr, Y_N: tl.constexpr,
            HAS_BIAS: tl.constexpr, MODE: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
            ACTIVATION: tl.constexpr = "", ATOMIC: tl.constexpr = False):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    k = tl.arange(0, BK)
    word = tl.arange(0, BK // 16)
    lane = tl.arange(0, 16)
    acc = tl.zeros((BM, BN), tl.float32)
    low_acc = tl.zeros((BM, BN), tl.float32)
    for tile in range(tl.cdiv(K, BK * SPLIT)):
        start = (tile * SPLIT + part) * BK
        kk = start + k
        a = tl.load(X + (m[:, None] // T) * X_B + (m[:, None] % T) * X_T + kk[None, :] * X_K,
                    (m[:, None] < M) & (kk[None, :] < K), other=0)
        w = tl.load(W + (start // 16 + word[:, None]) * N + n[None, :],
                    (start // 16 + word[:, None] < tl.cdiv(K, 16)) & (n[None, :] < N), other=0)
        c = ((w[:, None, :].to(tl.uint32) >> (lane[None, :, None] * 2)) & 3).reshape(BK, BN)
        b = ((c & 1).to(tl.int32) - (c >> 1).to(tl.int32))
        if MODE == "tf32x3":
            acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="tf32x3")
        elif MODE == "bf16x3" or MODE == "bf16x2":
            # Ternary B is exact in BF16. Three non-overlapping BF16 pieces
            # approximate A's full FP32 mantissa, with FP32 accumulation.
            af = a.to(tl.float32)
            hi = af.to(tl.bfloat16)
            rem = af - hi.to(tl.float32)
            mid = rem.to(tl.bfloat16)
            bb = b.to(tl.bfloat16)
            if MODE == "bf16x3":
                lo = (rem - mid.to(tl.float32)).to(tl.bfloat16)
                low_acc = tl.dot(lo, bb, low_acc)
            low_acc = tl.dot(mid, bb, low_acc)
            acc = tl.dot(hi, bb, acc)
        elif MODE == "fp16x2":
            # Experimental bounded-input arithmetic: scaling the residual
            # avoids FP16 subnormal loss. Not safe for arbitrary FP32 range.
            af = a.to(tl.float32)
            hi = af.to(tl.float16)
            lo = ((af - hi.to(tl.float32)) * 4096.0).to(tl.float16)
            bb = b.to(tl.float16)
            low_acc = tl.dot(lo, bb, low_acc)
            acc = tl.dot(hi, bb, acc)
        elif MODE == "bf16":
            acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
        elif MODE == "tf32":
            acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="tf32")
        else:
            acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)
    if MODE == "bf16x3" or MODE == "bf16x2":
        acc += low_acc
    elif MODE == "fp16x2":
        acc += low_acc * (1.0 / 4096.0)
    if ATOMIC:
        acc *= tl.load(S + n, n < N, other=0)[None, :]
        if HAS_BIAS:
            acc += tl.where(part == 0, tl.load(Bias + n, n < N, other=0)[None, :], 0.)
        tl.atomic_add(Y + (m[:, None] // T)*Y_B + (m[:, None] % T)*Y_T + n[None, :]*Y_N,
                      acc, (m[:, None] < M) & (n[None, :] < N), sem="relaxed")
    elif SPLIT == 1:
        acc *= tl.load(S + n, n < N, other=0)[None, :]
        if HAS_BIAS:
            acc += tl.load(Bias + n, n < N, other=0)[None, :]
        if ACTIVATION == "silu":
            acc = acc / (1.0 + tl.exp(-acc))
        tl.store(Y + (m[:, None] // T) * Y_B + (m[:, None] % T) * Y_T + n[None, :] * Y_N,
                 acc, (m[:, None] < M) & (n[None, :] < N))
    else:
        tl.store(Partial + part * M * N + m[:, None] * N + n[None, :],
                 acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _finish(P, S, Bias, Y, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr,
            Y_B: tl.constexpr, Y_T: tl.constexpr, Y_N: tl.constexpr,
            HAS_BIAS: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr,
            ACTIVATION: tl.constexpr = ""):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for p in tl.static_range(SPLIT):
        acc += tl.load(P + p * M * N + i, i < M * N, other=0)
    m, n = i // N, i % N
    acc *= tl.load(S + n, i < M * N, other=0)
    if HAS_BIAS:
        acc += tl.load(Bias + n, i < M * N, other=0)
    if ACTIVATION == "silu":
        acc = acc / (1.0 + tl.exp(-acc))
    tl.store(Y + (m // T) * Y_B + (m % T) * Y_T + n * Y_N, acc, i < M * N)


def configuration(m: int, n: int, k: int) -> tuple[int, int, int, int, int, int]:
    """BM, BN, BK, split-K, warps, stages. Offline-tuned defaults for SM120."""
    if m <= 32:
        return (32, 64, 32, 8, 4, 2)
    if m <= 64:
        return (64, 32, 32, 8, 4, 2)
    if k >= 2048:
        return (64, 64, 32, 8, 4, 2)
    if m <= 192 and n <= 2048:
        return (64, 32, 64 if n <= 1024 else 32, 8, 4, 2)
    return (64, 32 if n <= 1024 else 64, 32, 2 if n >= 4096 else 4, 4, 2)


def matmul(x: torch.Tensor, packed_t: torch.Tensor, scales: torch.Tensor,
           bias: torch.Tensor | None, k: int, *, conv: bool = False,
           mode: str = "bf16x3", config: tuple | None = None,
           out: torch.Tensor | None = None, activation: str = "", atomic: bool = False) -> torch.Tensor:
    """Linear [...,K] or pointwise convolution [B,K,T], including strided input.

    Preparation validates packed tensors. This hot path performs only metadata
    checks so it can run on non-default streams and under CUDA graph capture.
    """
    if not x.is_cuda or x.device != packed_t.device or x.dtype not in (torch.float32, torch.float16):
        raise ValueError("expected FP32/FP16 CUDA input on the weight device")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("packed kernels are inference-only; use torch.inference_mode()")
    if mode not in ("tf32x3", "bf16x3", "fp16", "bf16x2", "fp16x2", "bf16", "tf32", "fp16_rounded"):
        raise ValueError(f"unknown arithmetic mode {mode!r}")
    if activation not in ("", "silu"):
        raise ValueError(f"unknown activation {activation!r}")
    if atomic and activation:
        raise ValueError("atomic experiment cannot fuse nonlinear activation")
    if mode == "fp16" and x.dtype != torch.float16:
        raise ValueError("fp16 arithmetic requires explicitly FP16 inputs")
    n = scales.numel()
    if (k < 1 or n < 1 or scales.shape != (n,) or scales.dtype != torch.float32
            or scales.device != x.device or not scales.is_contiguous()
            or packed_t.shape != (triton.cdiv(k, 16), n) or packed_t.dtype != torch.int32
            or not packed_t.is_contiguous()):
        raise ValueError("expected contiguous packed int32 [ceil(K/16),N] and FP32 scales [N]")
    if bias is not None and (bias.shape != (n,) or bias.device != x.device
                             or bias.dtype != torch.float32 or not bias.is_contiguous()):
        raise ValueError("bias must be contiguous FP32 [N] on the input device")
    if conv:
        if x.ndim != 3 or x.shape[1] != k:
            raise ValueError(f"expected [B,{k},T] convolution input")
        b, _, t = x.shape
        xb, xk, xt = x.stride()
        shape = (b, n, t)
        yb, yt, yn = n * t, 1, t
    else:
        if x.ndim < 1 or x.shape[-1] != k:
            raise ValueError(f"expected [...,{k}] linear input")
        shape = (*x.shape[:-1], n)
        # Preserve the common NeMo [B,T,K] / transposed view without a copy.
        xx = x if x.ndim == 3 else x.reshape(1, -1, k)
        b, t, _ = xx.shape
        xb, xt, xk = xx.stride()
        x = xx
        yb, yt, yn = t * n, n, 1
    m = b * t
    if out is None:
        out = torch.empty(shape, device=x.device, dtype=x.dtype)
    elif out.shape != shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise ValueError("out must be contiguous with the correct shape, dtype, and device")
    elif out.numel() and any(out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
             for tensor in (x, packed_t, scales, bias) if tensor is not None):
        raise ValueError("out must not share storage with inputs, weights, scales, or bias")
    if not m:
        return out
    if m <= 4 and config is None:
        split = 4 if k > 2048 else 1
        partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32) if split > 1 else out
        with torch.cuda.device(x.device):
            _gemv[(triton.cdiv(n, 8), m, split)](
                x, packed_t, scales, bias if bias is not None else scales, out, partial,
                m, n, k, t, xb, xt, xk, yb, yt, yn, bias is not None, 8,
                max(16, triton.next_power_of_2(triton.cdiv(k, split))), split, ACTIVATION=activation, num_warps=4)
            if split > 1:
                _finish[(triton.cdiv(m * n, 256),)](
                    partial, scales, bias if bias is not None else scales, out,
                    m, n, t, yb, yt, yn, bias is not None, split, 256, ACTIVATION=activation)
        return out
    bm, bn, bk, split, warps, stages = config or configuration(m, n, k)
    if atomic:
        out.zero_()
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32) if split > 1 and not atomic else out
    with torch.cuda.device(x.device):
        _matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
            x, packed_t, scales, bias if bias is not None else scales, out, partial,
            m, n, k, t, xb, xt, xk, yb, yt, yn, bias is not None, mode,
            bm, bn, bk, split, ACTIVATION=activation, ATOMIC=atomic, num_warps=warps, num_stages=stages)
        if split > 1 and not atomic:
            _finish[(triton.cdiv(m * n, 256),)](
                partial, scales, bias if bias is not None else scales, out,
                m, n, t, yb, yt, yn, bias is not None, split, 256, ACTIVATION=activation)
    return out
