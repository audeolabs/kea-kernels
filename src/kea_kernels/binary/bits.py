"""1-bit weight codes: the export layout, checks and dequantization.

Layout (parakeet-binary-v1): uint8 [N, ceil(K/8)]; code j of a row is bit j % 8 of byte j // 8 (least significant bit
first); bit 1 = +1, bit 0 = -1; unused tail bits 0. Every row has one FP32 scale alpha, so W = alpha * sign.
Pure PyTorch; runs on the CPU.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

_SHIFTS = torch.arange(8, dtype=torch.uint8)


def packed_row_bytes(in_features: int) -> int:
    return math.ceil(in_features / 8)


def validate_weights(codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int) -> tuple[int, int]:
    """(N, K) after the checks prepare() must make; ValueError otherwise. Syncs if tensors are on GPU."""
    if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features < 1:
        raise ValueError(f"in_features must be a positive int, got {in_features!r}")
    if codes.dtype != torch.uint8 or codes.dim() != 2 or codes.shape[0] < 1 \
            or codes.shape[1] != packed_row_bytes(in_features):
        raise ValueError(f"codes must be uint8 [N, {packed_row_bytes(in_features)}], got {codes.dtype} {tuple(codes.shape)}")
    n = codes.shape[0]
    tail = 8 * codes.shape[1] - in_features
    if tail and (codes[:, -1] >> (8 - tail)).any():
        raise ValueError("invalid packed codes: nonzero tail padding")
    if scale.dtype != torch.float32 or scale.shape != (n,) or not torch.isfinite(scale).all():
        raise ValueError("scale must be finite FP32 [N]")
    if bias is not None and (bias.dtype != torch.float32 or bias.shape != (n,) or not torch.isfinite(bias).all()):
        raise ValueError("bias must be None or finite FP32 [N]")
    if not (codes.device == scale.device and (bias is None or bias.device == codes.device)):
        raise ValueError("codes, scale and bias must be on one device")
    return n, in_features


def unpack_bits(codes: Tensor, in_features: int) -> Tensor:
    """uint8 [N, ceil(K/8)] -> uint8 {0, 1} [N, K] (bit 1 = +1). Rejects nonzero tail bits."""
    if codes.dtype != torch.uint8 or codes.dim() != 2 or codes.shape[1] != packed_row_bytes(in_features):
        raise ValueError(f"expected uint8 [N, {packed_row_bytes(in_features)}], got {codes.dtype} {tuple(codes.shape)}")
    bits = ((codes[..., None] >> _SHIFTS.to(codes.device)) & 1).flatten(1)
    if bits[:, in_features:].any():
        raise ValueError("invalid packed codes: nonzero tail padding")
    return bits[:, :in_features].contiguous()


def unpack_signs(codes: Tensor, in_features: int) -> Tensor:
    """uint8 [N, ceil(K/8)] -> int8 {-1, +1} [N, K]."""
    return (2 * unpack_bits(codes, in_features).to(torch.int8) - 1).to(torch.int8)


def pack_signs(signs: Tensor) -> Tensor:
    """int8 {-1, +1} [N, K] -> packed codes uint8 [N, ceil(K/8)]."""
    if signs.dtype != torch.int8 or signs.dim() != 2 or ((signs != 1) & (signs != -1)).any():
        raise ValueError("expected 2-D int8 signs in {-1, +1}")
    rows, k = signs.shape
    bits = torch.zeros(rows, 8 * packed_row_bytes(k), dtype=torch.uint8, device=signs.device)
    bits[:, :k] = (signs == 1).to(torch.uint8)
    bits = bits.view(rows, -1, 8) << _SHIFTS.to(signs.device)
    return bits.sum(dim=2, dtype=torch.uint8)  # disjoint bits: sum == OR


def dense_weight(codes: Tensor, scale: Tensor, in_features: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Dequantized [N, K] = signs * scale[:, None] (exact in FP32 and FP64), on codes.device."""
    return unpack_signs(codes, in_features).to(dtype) * scale.to(device=codes.device, dtype=dtype)[:, None]


def resident_bytes(*tensors: Tensor | None) -> int:
    """Sum of distinct CUDA storages' full nbytes (a view counts its whole storage once)."""
    seen: dict[int, int] = {}
    for t in tensors:
        if t is None or not t.is_cuda:
            continue
        storage = t.untyped_storage()
        seen[storage.data_ptr()] = max(seen.get(storage.data_ptr(), 0), storage.nbytes())
    return sum(seen.values())
