"""The 1-bit layers at the ternary runtime's projection-provider seam.

The ternary runtime routes every quantized-layer call through one `projection` object, called by
ternary.transforms.encoder_transform and ternary.layer_fusions4.layer_fusions as

    provider(x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation="")

with these semantics (ternary.encoder4.ExpandedProjection.__call__):
    xx = x.transpose(1, 2) if conv else x           # conv: [B, K, T] -> [B, T, K] view
    z  = LayerNorm(norm)(xx) if norm else xx        # fused input normalization
    z  = z * sigmoid(z) if input_activation == "silu" else z
    y  = alpha * (W z) + bias                       # the quantized layer, [B, T, N]
    y  = y * sigmoid(y) if activation == "silu" else y
    return y.transpose(1, 2) if conv else y          # conv returns a [B, N, T] view
`components` is the number of INT8 residual activation parts (3 unless PartCount overrides it).

MODEL CONSTRUCTION (one copy of the weights): `load_binary_model` builds NeMo from the export's manifest and replaces
the 264 quantized modules by BinaryLinear / BinaryPointwiseConv1d (subclasses of the ternary PackedLinear /
PackedPointwiseConv1d, so the ternary transforms' type checks pass). Q, K and V of each block are prepared as ONE fused
[3072, 1024] layer; the three modules are row slices of it. A module's `packed_t` is a CPU int64 HANDLE [1, N]
(global row ids, never on the GPU); torch.cat of the Q/K/V handles in encoder_transform yields the fused handle, which
the provider resolves to the fused layer. `scale` is a host copy that the QKV concatenation needs and nothing computes
with. The 1-bit layer object holds the only GPU copy of codes and scales.

RUNTIMES: `enable_binary_optimizations(model, provider)` runs the ternary runtime's
enable_optimizations(model, encoder_storage="expanded") with ternary.encoder4.ExpandedProjection temporarily replaced
by a factory returning `provider` (the identical transform stack, fusions, decoder kernels and graph cache; only the
projection differs). This is the x1 stack. The x2 / x3 stacks are hybrid.hybrid_pipeline with the hybrid providers.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

QKV = ("linear_q", "linear_k", "linear_v")


@dataclass
class Entry:
    """One prepared 1-bit layer and the global handle range of its rows."""
    name: str
    layer: object
    kind: str            # "linear" or "conv"
    out_features: int
    in_features: int
    start: int           # first global row id
    members: tuple = ()  # for a fused QKV entry: the three module names


@dataclass
class Table:
    entries: list[Entry] = field(default_factory=list)
    next_row: int = 0

    def add(self, name, layer, kind, n, k, members=()) -> Entry:
        e = Entry(name, layer, kind, n, k, self.next_row, tuple(members))
        self.entries.append(e)
        self.next_row += n
        return e

    def resolve(self, start: int, count: int) -> tuple[Entry, int]:
        for e in self.entries:
            if e.start <= start < e.start + e.out_features:
                if start + count > e.start + e.out_features:
                    raise ValueError("handle range crosses layers")
                return e, start - e.start
        raise KeyError(f"no layer owns row handle {start}")

    def weight_bytes(self) -> int:
        return sum(e.layer.weight_bytes() for e in self.entries)


def _handle(start: int, n: int) -> Tensor:
    return torch.arange(start, start + n, dtype=torch.int64).view(1, n)


class _BinaryMixin:
    """Attributes the ternary transforms read: in_features, out_features, mode, packed_t, scale, bias."""

    def _setup(self, entry: Entry, offset: int, n: int, k: int, scale_cpu: Tensor, bias_cpu: Tensor | None):
        nn.Module.__init__(self)
        self.in_features, self.out_features, self.mode = k, n, "binary"
        self._entry, self._offset = entry, offset
        # Plain attributes (not buffers): never moved to the GPU, absent from state_dict.
        object.__setattr__(self, "packed_t", _handle(entry.start + offset, n))
        object.__setattr__(self, "scale", scale_cpu.detach().clone().contiguous())
        object.__setattr__(self, "bias", None if bias_cpu is None else bias_cpu.detach().clone().contiguous())
        self.eval()

    def _run(self, x: Tensor, conv: bool) -> Tensor:
        e = self._entry
        if self.out_features == e.out_features:
            return e.layer.conv(x) if conv else e.layer.linear(x)
        y = e.layer.conv(x) if conv else e.layer.linear(x)  # a member of a fused layer called alone: slice
        return y[:, self._offset:self._offset + self.out_features] if conv else \
            y[..., self._offset:self._offset + self.out_features]

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, layer={self._entry.name!r}"


def _packed_classes():
    from kea_kernels.ternary.runtime import PackedLinear, PackedPointwiseConv1d

    class BinaryLinear(_BinaryMixin, PackedLinear):
        def __init__(self, *a):
            self._setup(*a)

        def forward(self, x):
            return self._run(x, False)

    class BinaryPointwiseConv1d(_BinaryMixin, PackedPointwiseConv1d):
        def __init__(self, *a):
            self._setup(*a)

        def forward(self, x):
            return self._run(x, True)

    return BinaryLinear, BinaryPointwiseConv1d


class BinaryProjection:
    """The provider object handed to the ternary transforms (module docstring)."""

    def __init__(self, table: Table, names: dict[int, str] | None = None):
        self.table = table
        self._resolved: dict[tuple, tuple[Entry, int, int]] = {}
        self.names = names or {}

    @property
    def bytes(self) -> int:
        return self.table.weight_bytes()

    def _lookup(self, mod) -> tuple[Entry, int, int]:
        handle = mod.packed_t
        key = (handle.data_ptr(), tuple(handle.shape))
        hit = self._resolved.get(key)
        if hit is None:
            if handle.is_cuda or handle.dtype != torch.int64 or handle.dim() != 2 or handle.shape[0] != 1:
                raise TypeError("BinaryProjection serves BinaryLinear / BinaryPointwiseConv1d handles only")
            start, count = int(handle[0, 0]), handle.shape[1]
            if not torch.equal(handle[0], torch.arange(start, start + count)):
                raise ValueError("non-contiguous row handle")
            entry, offset = self.table.resolve(start, count)
            hit = (entry, offset, count)
            self._resolved[key] = hit
        return hit

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        if config is not None:
            raise ValueError("BinaryProjection takes no tile config")
        if activation not in ("", "silu") or input_activation not in ("", "silu"):
            raise ValueError(f"unsupported activation {activation!r} / {input_activation!r}")
        entry, offset, count = self._lookup(mod)
        layer = entry.layer
        if hasattr(layer, "project"):
            y = layer.project(x, conv=conv, components=components, activation=activation, norm=norm,
                              input_activation=input_activation)
        else:
            if conv and norm is None and not input_activation:
                y = layer.conv(x)
                if activation == "silu":
                    y = F.silu(y)
            else:
                z = x.transpose(1, 2) if conv else x
                if norm is not None:
                    z = F.layer_norm(z, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
                if input_activation == "silu":
                    z = F.silu(z)
                y = layer.linear(z)
                if activation == "silu":
                    y = F.silu(y)
                if conv:
                    y = y.transpose(1, 2)
        if count != entry.out_features:
            y = y[:, offset:offset + count] if conv else y[..., offset:offset + count]
        return y


class PartCount:
    """Wraps a projection provider and sets `components` (the INT8 activation part count) to n on EVERY call,
    including layer_fusions4's hard-coded components=3; every other argument, the stack and its fusions are unchanged.
    Attribute writes are forwarded to the wrapped provider for every name except the adapter's own (provider, n), so
    stack contexts that set provider state (row_fusion's defer_epilogue, no_defer_weights) reach the provider."""

    OWN = ("provider", "n")

    def __init__(self, provider, n: int):
        if n not in (1, 2, 3):
            raise ValueError("n must be 1, 2 or 3")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "n", n)

    def __setattr__(self, name, value):
        if name in PartCount.OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.provider, name, value)

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        return self.provider(x, mod, conv, self.n, activation, norm, config, input_activation)

    def __getattr__(self, name):  # .bytes, .table, ... of the wrapped provider
        return getattr(self.provider, name)


def load_binary_model(directory, manifest: dict, layer_cls, device="cuda", *, fuse_qkv: bool = True):
    """(NeMo model on `device`, Table) from a validated parakeet-binary-v1 export (formats.check_export).

    Non-quantized tensors are loaded FP16 -> FP32 with a strict key check. The 264 quantized modules hold no GPU
    tensors; `layer_cls.prepare` makes the only GPU copy of codes and scales (Q/K/V fused when fuse_qkv).
    """
    from safetensors.torch import load_file

    from kea_kernels import formats
    directory = Path(directory)
    device = torch.device(device)
    layers = manifest["quantized_layers"]
    model = formats.build_nemo(manifest, directory)
    tensors = load_file(directory / manifest["file"])
    BinaryLinear, BinaryPointwiseConv1d = _packed_classes()
    table = Table()
    pending = dict(layers)
    replacements = {}
    if fuse_qkv:
        blocks = sorted({n.rsplit(".", 1)[0] for n in layers if n.endswith(".self_attn.linear_q")})
        for block in blocks:
            names = [f"{block}.{q}" for q in QKV]
            shapes = [tuple(layers[n]["shape"]) for n in names]
            if len(set(shapes)) != 1 or any(layers[n]["kind"] != "linear" or layers[n]["bias"] for n in names):
                raise ValueError(f"{block}: Q/K/V must be equal-shape bias-free linear layers")
            n, k = shapes[0]
            codes = torch.cat([tensors.pop(f"{x}.codes") for x in names])
            scales = torch.cat([tensors.pop(f"{x}.scale") for x in names])
            layer = layer_cls.prepare(codes, scales, None, k, device)
            entry = table.add(f"{block}.qkv", layer, "linear", 3 * n, k, names)
            for i, x in enumerate(names):
                replacements[x] = BinaryLinear(entry, i * n, n, k, scales[i * n:(i + 1) * n], None)
                del pending[x]
    for name, info in pending.items():
        n, k = info["shape"]
        codes, scale = tensors.pop(f"{name}.codes"), tensors.pop(f"{name}.scale")
        bias = tensors.pop(f"{name}.bias") if info["bias"] else None
        kind = "conv" if info["kind"] == "pointwise_conv1d" else "linear"
        layer = layer_cls.prepare(codes, scale, bias, k, device)
        entry = table.add(name, layer, kind, n, k)
        cls = BinaryPointwiseConv1d if kind == "conv" else BinaryLinear
        replacements[name] = cls(entry, 0, n, k, scale, bias)
    for name, module in replacements.items():
        original = model.get_submodule(name)
        if tuple(original.weight.shape[:2]) != (module.out_features, module.in_features):
            raise ValueError(f"invalid shape for {name}")
        model.set_submodule(name, module)
    state = {key: value.float() if value.is_floating_point() else value for key, value in tensors.items()}
    model.load_state_dict(state, strict=True)  # the binary modules hold no parameters or buffers
    model = model.to(device=device).eval()
    return model, table


def module_names(model) -> dict[int, str]:
    return {id(m): n for n, m in model.named_modules()}


@contextlib.contextmanager
def _replace_expanded_projection(provider):
    """ternary.optimized.enable_optimizations imports ExpandedProjection from ternary.encoder4 at call time; swap that
    module attribute for a factory returning `provider` during the call."""
    import kea_kernels.ternary.encoder4 as encoder4
    original = encoder4.ExpandedProjection

    def factory(backend="triton", config=None):
        if backend != "triton_col_tuned" or config is not None:
            raise AssertionError(f"unexpected ExpandedProjection({backend!r}, {config!r})")
        return provider

    encoder4.ExpandedProjection = factory
    try:
        yield
    finally:
        encoder4.ExpandedProjection = original


def enable_binary_optimizations(model, provider, *, max_graphs: int = 4):
    """The ternary enable_optimizations(model, encoder_storage="expanded") with `provider` as the projection."""
    from kea_kernels.ternary.optimized import enable_optimizations
    with _replace_expanded_projection(provider):
        return enable_optimizations(model, max_graphs=max_graphs, decoder=True, position_dot=True,
                                    encoder_storage="expanded")
