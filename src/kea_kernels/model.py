"""load / transcribe / close: the three models on their kernels, straight from their Hugging Face downloads.

Modes (the activation precision of the 264 quantized encoder layers; weights stay at 1 or 2 bits):
  Kea-B (1-bit, parakeet-binary-v1)
    x1  one INT8 activation part per frame ("i8x1"): the ternary expanded stack with the 1-bit layers as its
        projection (provider.enable_binary_optimizations), every call at one part. Kea-B 0.1's default.
    x2  two residual INT8 parts: the hybrid stack (deferred row epilogues, exact integer split-K) with the 2-part
        layers. Kea-B 0.0's default.
    x3  three residual INT8 parts: the same hybrid stack at three parts.
  Kea-T (ternary, parakeet-ternary-v1)
    x2  the expanded stack (an exact INT8 copy of the ternary codes, about 604 MB more VRAM) at two parts. Default.
    x3  the expanded stack at three parts.
    compact  the ternary codes stay 2-bit on the GPU, unpacked inside the kernels (three parts).
Every mode decodes with NeMo's greedy TDT decoder, fused, as conditional CUDA graphs, at batch 1 in strict FP32.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from kea_kernels import formats

SAMPLE_RATE = 16000
# Inputs shorter than 30 ms are zero-padded to 30 ms, as in the published evaluations.
MIN_DECODE_SAMPLES = 480
SUPPORTED_CAPABILITY = (12, 0)


@dataclass
class Transcript:
    text: str
    tokens: list[int]        # SentencePiece token IDs
    timestamps: list[int]    # encoder frame of each token (80 ms per frame)


@dataclass
class Model:
    """A loaded model. `nemo` is the underlying NeMo EncDecRNNTBPEModel, transformed for inference: do not train,
    move or save it; call close() (or use the Model as a context manager) to release its CUDA graphs."""
    nemo: object
    name: str
    kind: str
    mode: str
    export: str
    device: torch.device
    _stack: contextlib.ExitStack = field(repr=False)
    _featurizer: dict = field(repr=False, default_factory=dict)
    closed: bool = False

    def transcribe(self, audio, **kwargs):
        return transcribe(self, audio, **kwargs)

    def close(self) -> None:
        close(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        close(self)


def check_device(device) -> torch.device:
    """The CUDA device, or ValueError: the kernels are compiled and tuned for SM120 (RTX 5090) only."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("kea-kernels runs on NVIDIA GPUs only (device must be 'cuda' or 'cuda:N')")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available to PyTorch (no NVIDIA driver or a CPU-only torch build)")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if capability != SUPPORTED_CAPABILITY:
        name = torch.cuda.get_device_name(device)
        raise ValueError(f"this runtime targets NVIDIA SM120 (RTX 5090); {device} is {name} (SM{capability[0]}"
                         f"{capability[1]}). The kernels use SM120 instructions and tile tables, and no other GPU has "
                         "been tested.")
    return device


def load(path, device="cuda", mode: str | None = None, *, verify: bool = True) -> Model:
    """Load a downloaded Kea-B 0.1, Kea-B or Kea-T directory onto `device` in `mode` (default: the model's own).

    verify: check the SHA-256 of every file the manifest lists (a few seconds)."""
    info = formats.describe(path)
    modes = formats.BINARY_MODES if info.kind == "binary" else formats.TERNARY_MODES
    mode = info.default_mode if mode is None else mode
    if mode not in modes:
        raise ValueError(f"{info.name} supports modes {modes}, got {mode!r}")
    manifest = formats.check_export(info.export, info.kind, verify_hashes=verify)
    device = check_device(device)
    from kea_kernels import _nvrtc
    _nvrtc.setup()  # load NVRTC globally first: the kernels and NeMo's CUDA-graph decoder both need it
    torch.cuda.set_device(device)
    stack = contextlib.ExitStack()
    with torch.inference_mode():
        try:
            if info.kind == "binary":
                nemo = _load_binary(info.export, manifest, device, mode, stack)
            else:
                nemo = _load_ternary(info.export, manifest, device, mode, stack)
        except BaseException:
            stack.close()
            raise
        featurizer = nemo.preprocessor.featurizer
        saved = {k: getattr(featurizer, k) for k in ("dither", "pad_to")}
        featurizer.dither, featurizer.pad_to = 0.0, 0
        torch.cuda.synchronize(device)
    return Model(nemo, info.name, info.kind, mode, str(info.export), device, stack, saved)


def _load_binary(export: Path, manifest: dict, device, mode: str, stack: contextlib.ExitStack):
    from kea_kernels.binary import provider as P
    from kea_kernels.binary.layer import Int8MmaLayer
    from kea_kernels.binary.layer_parts import LAYERS
    parts = int(mode[1])
    layer_cls = Int8MmaLayer if parts == 3 else LAYERS[parts]
    model, table = P.load_binary_model(export, manifest, layer_cls, device)
    model.eval()
    model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
    names = P.module_names(model)
    if parts == 1:
        from kea_kernels.ternary.optimized import disable_optimizations
        provider = P.PartCount(P.BinaryProjection(table, names), 1)
        P.enable_binary_optimizations(model, provider, max_graphs=4)
        stack.callback(disable_optimizations, model)
    elif parts == 2:
        from kea_kernels.binary.hybrid_x2 import HybridProjectionX2, exact_hybrid_pipeline_x2
        provider = P.PartCount(HybridProjectionX2(table, names), 2)
        stack.enter_context(exact_hybrid_pipeline_x2(model, provider))
    else:
        from kea_kernels.binary.hybrid_exact import HybridProjectionExact, exact_hybrid_pipeline
        provider = HybridProjectionExact(table, names)
        stack.enter_context(exact_hybrid_pipeline(model, provider))
    return model


def _load_ternary(export: Path, manifest: dict, device, mode: str, stack: contextlib.ExitStack):
    from kea_kernels.ternary.optimized import disable_optimizations, enable_optimizations
    from kea_kernels.ternary.runtime import load_packed
    model = load_packed(export, manifest, device)
    if mode == "compact":
        enable_optimizations(model, encoder_storage="packed")
    else:
        import kea_kernels.ternary.encoder4 as e4
        from kea_kernels.binary.provider import PartCount
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
        parts = int(mode[1])
        original = e4.ExpandedProjection

        def factory(backend="triton", config=None):
            inner = original(backend, config)
            return PartCount(inner, parts) if parts != 3 else inner

        e4.ExpandedProjection = factory
        try:
            enable_optimizations(model, max_graphs=4, encoder_storage="expanded")
        finally:
            e4.ExpandedProjection = original
    model.eval()
    stack.callback(disable_optimizations, model)
    return model


def close(model: Model) -> None:
    """Release the CUDA graphs and undo the inference transforms. Idempotent."""
    if model.closed:
        return
    with torch.inference_mode():
        torch.cuda.synchronize(model.device)
        model._stack.close()
        for k, v in model._featurizer.items():
            setattr(model.nemo.preprocessor.featurizer, k, v)
    model.closed = True


# --- transcription ------------------------------------------------------------------------------------------------

def read_audio(path) -> np.ndarray:
    """float32 mono samples of a 16 kHz audio file; multichannel files are averaged to mono."""
    import soundfile
    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: {rate} Hz; the models take 16 kHz audio "
                         f"(convert with: ffmpeg -i {path} -ar 16000 -ac 1 out.wav)")
    return audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]


def _as_clip(item) -> np.ndarray:
    if isinstance(item, (str, os.PathLike)):
        clip = read_audio(item)
    else:
        clip = item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
        if clip.ndim != 1:
            raise ValueError(f"in-memory audio must be 1-D mono 16 kHz samples, got shape {clip.shape}")
        clip = clip.astype(np.float32, copy=False)
    if len(clip) < MIN_DECODE_SAMPLES:
        clip = np.pad(clip, (0, MIN_DECODE_SAMPLES - len(clip)))
    return clip


@contextlib.contextmanager
def _strict_fp32(device: torch.device):
    """Autocast off and TF32 off (matmul and cuDNN) for the duration; previous settings restored."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        with torch.autocast(device.type, enabled=False), torch.backends.cudnn.flags(
                enabled=True, benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.set_float32_matmul_precision(precision)


def _transcribe_one(model: Model, clip: np.ndarray) -> Transcript:
    nemo = model.nemo
    audio = torch.from_numpy(clip).unsqueeze(0).to(model.device)
    length = torch.tensor([len(clip)], dtype=torch.long, device=model.device)
    enc, enc_len = nemo.forward(input_signal=audio, input_signal_length=length)
    hyps = nemo.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                         return_hypotheses=True)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    h = hyps[0]
    seq = getattr(h, "y_sequence", None)
    tokens = [int(t) for t in (seq.tolist() if hasattr(seq, "tolist") else (seq or []))]
    ts = getattr(h, "timestamp", None)
    stamps = [int(t) for t in ts.tolist()] if hasattr(ts, "tolist") else [int(t) for t in (ts or [])]
    return Transcript(h.text if hasattr(h, "text") else str(h), tokens, stamps)


def transcribe(model: Model, audio, *, return_tokens: bool = False):
    """Greedy TDT transcripts (punctuated, cased) at batch 1.

    audio: one item or a list of items, each a path to a 16 kHz audio file or a 1-D float array / tensor of 16 kHz
    mono samples in [-1, 1]. Returns a str per item (a Transcript with token IDs and frame timestamps when
    return_tokens), and a list in input order for a list.

    Each input is decoded in a single pass with full attention; nothing is chunked. The models were trained on
    1-30 s utterances, so split long recordings (ideally at pauses) into segments of about 30 s or less."""
    if model.closed:
        raise ValueError("the model is closed")
    single = not isinstance(audio, (list, tuple))
    items = [audio] if single else list(audio)
    out = []
    with torch.inference_mode(), _strict_fp32(model.device):
        model.nemo.eval()
        for item in items:
            result = _transcribe_one(model, _as_clip(item))
            out.append(result if return_tokens else result.text)
    return out[0] if single else out


def decoder_graphs_mode(model: Model) -> str:
    """NeMo's TDT decoder CUDA-graph mode in use ("full_graph" expected; anything else means NVRTC or cuda-python was
    unusable and NeMo fell back to a slower decoder loop)."""
    comp = model.nemo.decoding.decoding.decoding_computer
    return str(getattr(comp, "cuda_graphs_mode", None))
