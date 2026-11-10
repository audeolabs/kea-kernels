"""Load a parakeet-ternary-v1 export directly into packed, inference-only NeMo modules."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .kernels import matmul


class PackedLinear(nn.Module):
    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, in_features: int,
                 bias: torch.Tensor | None = None, *, mode: str = "bf16x3"):
        super().__init__()
        from kea_kernels.formats import unpack_ternary

        if in_features < 1 or packed.ndim != 2 or packed.shape[0] < 1:
            raise ValueError("weight dimensions must be positive")
        if packed.device != scale.device or (bias is not None and bias.device != packed.device):
            raise ValueError("codes, scales, and bias must be on the same device")
        # Validate on CPU once, including unused 11 codes and tail bits.
        unpack_ternary(packed, in_features)
        n = packed.shape[0]
        if scale.shape != (n,) or scale.dtype != torch.float32 or not torch.isfinite(scale).all():
            raise ValueError("scale must be finite FP32 [N]")
        if bias is not None and (bias.shape != (n,) or bias.dtype != torch.float32 or not torch.isfinite(bias).all()):
            raise ValueError("bias must be finite FP32 [N]")
        if mode not in ("tf32x3", "bf16x3", "fp16"):
            raise ValueError(f"unknown mode {mode!r}")
        self.in_features, self.out_features = in_features, n
        self.mode = mode
        # Four little-endian export bytes form one uint32 decode word.
        padded = torch.nn.functional.pad(packed, (0, (-packed.shape[1]) % 4))
        self.register_buffer("packed_t", padded.contiguous().view(torch.int32).t().contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.register_buffer("bias", bias.contiguous() if bias is not None else None)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return matmul(x, self.packed_t, self.scale, self.bias, self.in_features, mode=self.mode)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, mode={self.mode!r}"


class PackedPointwiseConv1d(PackedLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return matmul(x, self.packed_t, self.scale, self.bias, self.in_features, conv=True, mode=self.mode)


def load_packed(directory: str | Path, manifest: dict, device: str | torch.device = "cuda", *, mode: str = "bf16x3"):
    """Construct NeMo from a validated export (formats.check_export), replacing all 264 ternary layers.

    No requantization and no .nemo download. The initial NeMo constructor uses CPU weights, which are discarded as
    modules are replaced. No dense ternary weight is ever moved to the GPU. The decoder and remaining layers are FP32.
    """
    from safetensors.torch import load_file

    from kea_kernels import formats
    if mode not in ("tf32x3", "bf16x3"):
        raise ValueError("the model loader requires a high-accuracy FP32-input mode (tf32x3 or bf16x3)")
    device = torch.device(device)
    directory = Path(directory)
    layers = manifest["quantized_layers"]
    model = formats.build_nemo(manifest, directory)
    tensors = load_file(directory / manifest["file"])
    packed_state = {}
    for name, layer in layers.items():
        original = model.get_submodule(name)
        n, k = layer["shape"]
        if tuple(original.weight.shape[:2]) != (n, k):
            raise ValueError(f"invalid shape for {name}")
        cls = {"linear": PackedLinear, "pointwise_conv1d": PackedPointwiseConv1d}[layer["kind"]]
        module = cls(tensors.pop(f"{name}.codes"), tensors.pop(f"{name}.scale"), k,
                     tensors.pop(f"{name}.bias") if layer["bias"] else None, mode=mode)
        model.set_submodule(name, module)
        packed_state.update((f"{name}.{key}", value) for key, value in module.state_dict().items())
    state = {key: value.float() if value.is_floating_point() else value for key, value in tensors.items()}
    state.update(packed_state)
    model.load_state_dict(state, strict=True)
    return model.to(device=device).eval()
