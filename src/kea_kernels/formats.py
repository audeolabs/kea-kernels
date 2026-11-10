"""The export formats the kernels read, their checks, and which model a downloaded directory holds.

Two export formats, one directory each (`export.safetensors`, `manifest.json`, `tokenizer/`):
  * "parakeet-binary-v1" (Kea-B): 264 quantized layers as `<name>.codes` uint8 [N, ceil(K/8)] (1 bit per weight,
    see binary.bits) plus `<name>.scale` FP32 [N];
  * "parakeet-ternary-v1" (Kea-T): `<name>.codes` uint8 [N, ceil(K/4)] (2 bits per weight: 00 = 0, 01 = +1,
    10 = -1, code j at bits 2 * (j % 4) of byte j // 4) plus `<name>.scale` FP32 [N].
Every other tensor is stored as FP16 (FP32 for the feature window and filterbank) and loaded as FP32.

Downloaded model directories:
  * audeolabs/kea-b-0.1: `manifest.json` is the 8-bit bundle ("keab-8bit-v1"); the kernels read its `compat/`
    directory, a parakeet-binary-v1 export of the same weights (its manifest records the bundle under extra.keab);
  * audeolabs/kea-b: a parakeet-binary-v1 export;
  * audeolabs/kea-t: a parakeet-ternary-v1 export.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

BINARY_FORMAT = "parakeet-binary-v1"
TERNARY_FORMAT = "parakeet-ternary-v1"
BUNDLE_FORMAT = "keab-8bit-v1"
FILE = "export.safetensors"
MANIFEST = "manifest.json"
COMPAT_DIR = "compat"
TOKENIZER_DIR = "tokenizer"
# config.tokenizer key -> file name in the export's tokenizer directory
TOKENIZER_FILES = {"model_path": "tokenizer.model", "vocab_path": "vocab.txt", "spe_tokenizer_vocab": "tokenizer.vocab"}
MODEL_TARGET = "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"

LAYERS = 24
PER_LAYER = ("feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1", "feed_forward2.linear2",
             "self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v", "self_attn.linear_out",
             "self_attn.linear_pos", "conv.pointwise_conv1", "conv.pointwise_conv2")
EXPECTED_MODULES = LAYERS * len(PER_LAYER)  # 264
QUANTIZED_PATTERN = re.compile(
    r"encoder\.layers\.\d+\.(feed_forward[12]\.linear[12]|self_attn\.linear_(q|k|v|out|pos)"
    r"|conv\.pointwise_conv[12])")

# Modes per export kind: the first is the default (the mode the model card's kernel numbers were measured in).
BINARY_MODES = ("x2", "x3", "x1")
TERNARY_MODES = ("x2", "x3", "compact")

_TERNARY_SHIFTS = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# --- 2-bit ternary codes ------------------------------------------------------------------------------------------

def pack_ternary(codes: Tensor) -> Tensor:
    """int8 [out, in] in {-1, 0, 1} -> uint8 [out, ceil(in/4)]; 0->00, +1->01, -1->10, code j at bits 2*(j%4)."""
    if codes.dtype != torch.int8 or codes.dim() != 2 or ((codes < -1) | (codes > 1)).any():
        raise ValueError("expected 2-D int8 codes in {-1, 0, 1}")
    rows, n = codes.shape
    fields = torch.zeros(rows, 4 * math.ceil(n / 4), dtype=torch.uint8, device=codes.device)
    fields[:, :n] = (codes == 1).to(torch.uint8) | ((codes == -1).to(torch.uint8) << 1)
    fields = fields.view(rows, -1, 4) << _TERNARY_SHIFTS.to(codes.device)
    return fields[..., 0] | fields[..., 1] | fields[..., 2] | fields[..., 3]


def unpack_ternary(packed: Tensor, in_features: int) -> Tensor:
    """Inverse of pack_ternary; rejects the unused 11 pattern and nonzero tail padding."""
    if packed.dtype != torch.uint8 or packed.dim() != 2 or packed.shape[1] != math.ceil(in_features / 4):
        raise ValueError(f"expected uint8 [out, {math.ceil(in_features / 4)}], got {packed.dtype} {tuple(packed.shape)}")
    fields = ((packed[..., None] >> _TERNARY_SHIFTS.to(packed.device)) & 3).flatten(1)
    if (fields == 3).any() or fields[:, in_features:].any():
        raise ValueError("invalid packed codes")
    fields = fields[:, :in_features]
    return (fields == 1).to(torch.int8) - (fields == 2).to(torch.int8)


# --- exports ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelDir:
    """What a directory holds: the export the kernels read and its kind."""
    name: str            # "kea-b-0.1", "kea-b" or "kea-t" (by format; any export of the same layout works)
    kind: str            # "binary" or "ternary"
    export: Path         # the parakeet-*-v1 export directory
    default_mode: str


def read_manifest(directory: Path) -> dict:
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise ValueError(f"{directory}: no {MANIFEST} (expected a downloaded Kea-B or Kea-T model directory)")
    return json.loads(path.read_text())


def describe(directory) -> ModelDir:
    """Which model `directory` holds (a downloaded audeolabs/kea-b-0.1, kea-b or kea-t, or an export inside one)."""
    directory = Path(directory).expanduser().resolve()
    manifest = read_manifest(directory)
    fmt = manifest.get("format")
    if fmt == BUNDLE_FORMAT:
        compat = directory / COMPAT_DIR
        if not (compat / MANIFEST).is_file():
            raise ValueError(f"{directory}: a {BUNDLE_FORMAT} bundle without its {COMPAT_DIR}/ export; download the "
                             "whole repository (hf download audeolabs/kea-b-0.1 --local-dir ...)")
        inner = describe(compat)
        if inner.kind != "binary":
            raise ValueError(f"{compat}: expected a {BINARY_FORMAT} export")
        bundle = (read_manifest(compat).get("extra") or {}).get("keab") or {}
        if bundle.get("bundle_sha256") != manifest.get("sha256"):
            raise ValueError(f"{compat}: its manifest does not record this directory's bundle")
        return inner
    if fmt == BINARY_FORMAT:
        keab = (manifest.get("extra") or {}).get("keab")
        if keab:  # Kea-B 0.1 (its compat export): measured and published in i8x1
            return ModelDir("kea-b-0.1", "binary", directory, "x1")
        return ModelDir("kea-b", "binary", directory, "x2")
    if fmt == TERNARY_FORMAT:
        return ModelDir("kea-t", "ternary", directory, "x2")
    raise ValueError(f"{directory}: unsupported format {fmt!r} (expected {BINARY_FORMAT}, {TERNARY_FORMAT} or "
                     f"{BUNDLE_FORMAT})")


def check_export(directory, kind: str, *, verify_hashes: bool = True) -> dict:
    """The manifest of an intact parakeet-binary-v1 / parakeet-ternary-v1 export of the 264 Parakeet layers.

    Checks the format (the manifest's and the safetensors metadata's), the SHA-256 of every listed file (unless
    verify_hashes is False), the 264 quantized layer names and kinds, and, without loading the tensors, each layer's
    codes / scale / bias dtypes and shapes. Raises ValueError otherwise."""
    from safetensors import safe_open
    directory = Path(directory)
    manifest = read_manifest(directory)
    fmt = {"binary": BINARY_FORMAT, "ternary": TERNARY_FORMAT}[kind]
    if manifest.get("format") != fmt:
        raise ValueError(f"{directory}: export format {manifest.get('format')!r}, expected {fmt!r}")
    if manifest.get("file") != FILE or FILE not in manifest.get("files", {}):
        raise ValueError(f"{directory}: manifest does not list {FILE}")
    if verify_hashes:
        for rel, digest in manifest["files"].items():
            path = directory / rel
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"{path} does not match the manifest SHA-256")
    cfg = manifest.get("config") or {}
    if cfg.get("target") != MODEL_TARGET:
        raise ValueError(f"{directory}: unexpected model target {cfg.get('target')!r}")
    layers = manifest.get("quantized_layers") or {}
    if len(layers) != EXPECTED_MODULES or any(not QUANTIZED_PATTERN.fullmatch(n) for n in layers):
        raise ValueError(f"{directory}: expected the {EXPECTED_MODULES} Parakeet quantized layers, got {len(layers)}")
    per_byte = 8 if kind == "binary" else 4
    with safe_open(directory / FILE, framework="pt") as f:
        if (f.metadata() or {}).get("format") != fmt:
            raise ValueError(f"{directory}: safetensors metadata format is not {fmt!r}")
        keys = set(f.keys())
        for name, info in layers.items():
            if info.get("kind") not in ("linear", "pointwise_conv1d"):
                raise ValueError(f"{name}: unknown kind {info.get('kind')!r}")
            n, k = info["shape"]
            if f"{name}.weight" in keys:
                raise ValueError(f"{name}: dense weight present in a quantized export")
            codes, scale = f.get_slice(f"{name}.codes"), f.get_slice(f"{name}.scale")
            if codes.get_dtype() != "U8" or codes.get_shape() != [n, math.ceil(k / per_byte)]:
                raise ValueError(f"{name}: codes must be uint8 [{n}, {math.ceil(k / per_byte)}]")
            if scale.get_dtype() != "F32" or scale.get_shape() != [n]:
                raise ValueError(f"{name}: scale must be FP32 [{n}]")
            if bool(info.get("bias")) != (f"{name}.bias" in keys):
                raise ValueError(f"{name}: bias presence differs from the manifest")
    return manifest


def build_nemo(manifest: dict, directory: Path):
    """An FP32 NeMo EncDecRNNTBPEModel from the manifest's config and the export's own tokenizer (no .nemo file)."""
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf, open_dict
    directory = Path(directory)
    cfg = OmegaConf.create(manifest["config"])
    with open_dict(cfg):
        cfg.tokenizer.dir = str(directory / TOKENIZER_DIR)
        for key, filename in TOKENIZER_FILES.items():
            cfg.tokenizer[key] = str(directory / TOKENIZER_DIR / filename)
        for ds in ("train_ds", "validation_ds", "test_ds"):
            if ds in cfg:
                cfg[ds] = None
    return EncDecRNNTBPEModel(cfg=cfg)
