"""kea-kernels: the GPU kernels and a minimal runtime for Audeo Labs' Kea-B (1-bit) and Kea-T (ternary) speech
recognition models.

    import kea_kernels
    model = kea_kernels.load("kea-b-0.1")           # a downloaded model directory; mode = the model's default
    print(kea_kernels.transcribe(model, "clip.wav"))
    kea_kernels.close(model)
"""
from kea_kernels.model import Model, Transcript, close, decoder_graphs_mode, load, transcribe

__version__ = "0.1.0"
__all__ = ["Model", "Transcript", "load", "transcribe", "close", "decoder_graphs_mode", "__version__"]
