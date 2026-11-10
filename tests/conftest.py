"""Shared fixtures. GPU tests are marked `gpu` and skipped unless an SM120 GPU (RTX 5090) is visible."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch


def sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0)


def pytest_collection_modifyitems(config, items):
    if sm120():
        return
    skip = pytest.mark.skip(reason="needs an NVIDIA SM120 GPU (RTX 5090)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Every test compiles into its own cubin cache (never the user's)."""
    monkeypatch.setenv("KEA_KERNELS_CACHE", str(tmp_path / "kernel-cache"))


def write_fake_export(directory: Path, kind: str, *, n: int = 8, k: int = 16, extra: dict | None = None) -> Path:
    """A structurally valid parakeet-*-v1 export with the 264 layer names at a tiny shape (no NeMo needed)."""
    from safetensors.torch import save_file
    from kea_kernels import formats
    directory.mkdir(parents=True, exist_ok=True)
    per_byte = 8 if kind == "binary" else 4
    layers, tensors = {}, {}
    for layer in range(formats.LAYERS):
        for suffix in formats.PER_LAYER:
            name = f"encoder.layers.{layer}.{suffix}"
            layers[name] = {"kind": "pointwise_conv1d" if "pointwise" in suffix else "linear", "shape": [n, k],
                            "bias": False}
            tensors[f"{name}.codes"] = torch.zeros(n, math.ceil(k / per_byte), dtype=torch.uint8)
            tensors[f"{name}.scale"] = torch.ones(n, dtype=torch.float32)
    fmt = formats.BINARY_FORMAT if kind == "binary" else formats.TERNARY_FORMAT
    save_file(tensors, str(directory / formats.FILE), metadata={"format": fmt})
    (directory / "tokenizer").mkdir(exist_ok=True)
    (directory / "tokenizer" / "vocab.txt").write_text("a\n")
    files = {rel: formats.sha256_file(directory / rel) for rel in (formats.FILE, "tokenizer/vocab.txt")}
    manifest = {"format": fmt, "file": formats.FILE, "sha256": files[formats.FILE], "files": files,
                "config": {"target": formats.MODEL_TARGET}, "quantized_layers": layers, "extra": extra or {}}
    (directory / formats.MANIFEST).write_text(json.dumps(manifest))
    return directory
