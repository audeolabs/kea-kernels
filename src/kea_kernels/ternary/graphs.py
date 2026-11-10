"""Bounded, opt-in encoder graph caching for repeated inference shapes.

The NeMo module tree and state_dict keys are unchanged. Both dense and packed
models can use this wrapper. Shapes beyond the cache limit run eagerly.
"""
from __future__ import annotations

import torch


class EncoderGraphCache:
    def __init__(self, encoder, max_graphs: int = 4):
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        self.encoder = encoder
        self.original_forward = encoder.forward
        self.max_graphs = max_graphs
        self.entries = {}

    def __call__(self, audio_signal, length=None, **kwargs):
        # Streaming caches and training retain the original NeMo semantics.
        if (self.encoder.training or torch.is_grad_enabled() or not audio_signal.is_cuda
                or length is None or length.device != audio_signal.device or kwargs
                or torch.is_autocast_enabled() or torch.cuda.is_current_stream_capturing()):
            return self.original_forward(audio_signal=audio_signal, length=length, **kwargs)
        device = audio_signal.device
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            key = (tuple(audio_signal.shape), audio_signal.dtype, device,
                   tuple(length.shape), length.dtype, stream.cuda_stream,
                   torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            if key not in self.entries:
                if len(self.entries) >= self.max_graphs:
                    return self.original_forward(audio_signal=audio_signal, length=length)
                static_audio = torch.empty_like(audio_signal, memory_format=torch.contiguous_format)
                static_length = torch.empty_like(length)
                static_audio.copy_(audio_signal)
                static_length.copy_(length)
                warm = torch.cuda.Stream(device=device)
                warm.wait_stream(stream)
                with torch.cuda.stream(warm):
                    for _ in range(3):
                        self.original_forward(audio_signal=static_audio, length=static_length)
                stream.wait_stream(warm)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=warm):
                    outputs = self.original_forward(audio_signal=static_audio, length=static_length)
                # NeMo may extend positional buffers for a later, longer shape.
                # Captured external pointers must keep their original storage.
                keepalive = tuple(self.encoder.parameters()) + tuple(self.encoder.buffers())
                # Transfer ownership of hoisted positional constants to this
                # graph, so clearing the graph cache releases old shapes too.
                for module in self.encoder.modules():
                    constants = getattr(module, "_packed_positional_capture", None)
                    if constants:
                        keepalive += tuple(constants)
                        constants.clear()
                self.entries[key] = (graph, static_audio, static_length, outputs, keepalive)
            graph, static_audio, static_length, outputs, _ = self.entries[key]
            static_audio.copy_(audio_signal)
            static_length.copy_(length)
            graph.replay()
            # The caller owns its result: later replays must not mutate it.
            return tuple(t.clone() for t in outputs)


def enable_encoder_graphs(model, max_graphs: int = 4) -> EncoderGraphCache:
    """Cache up to max_graphs exact shape/stream combinations; no audio padding.

    Enable after loading and placing the model on its final CUDA device. Keep
    model weights and device unchanged while the cache is active. For changed
    weights/device, disable the cache first and enable a new one afterwards.
    """
    if isinstance(model.encoder.forward, EncoderGraphCache) or isinstance(model.forward, PipelineGraphCache):
        raise ValueError("encoder graphs are already enabled")
    cache = EncoderGraphCache(model.encoder, max_graphs)
    model.encoder.forward = cache
    return cache


def disable_encoder_graphs(model) -> None:
    """Release cached graphs after pending CUDA work completes."""
    cache = model.encoder.forward
    if not isinstance(cache, EncoderGraphCache):
        return
    devices = {entry[1].device for entry in cache.entries.values()}
    for device in devices:
        torch.cuda.synchronize(device)
    model.encoder.forward = cache.original_forward
    cache.entries.clear()


class PipelineGraphCache(EncoderGraphCache):
    """The same exact-shape cache around raw-audio preprocessing + encoder."""

    def __init__(self, model, max_graphs=4):
        super().__init__(model, max_graphs)
        self.model_forward = self.original_forward
        self.original_forward = lambda audio_signal, length: self.model_forward(
            input_signal=audio_signal, input_signal_length=length)

    def __call__(self, input_signal=None, input_signal_length=None, **kwargs):
        frontend = getattr(getattr(self.encoder, "preprocessor", None), "featurizer", None)
        if (input_signal is None or input_signal_length is None or kwargs
                or getattr(frontend, "pad_to", 0) != 0 or getattr(frontend, "dither", 0) != 0):
            return self.model_forward(input_signal=input_signal, input_signal_length=input_signal_length, **kwargs)
        return super().__call__(audio_signal=input_signal, length=input_signal_length)


def enable_pipeline_graphs(model, max_graphs=4):
    """Capture deterministic raw-audio frontend and encoder; decoder stays NeMo.

    Use in eval mode with frontend dither=0 and pad_to=0. Disable before changing
    model weights, frontend settings, device, or dtype. Encoder and pipeline
    caches are mutually exclusive to avoid nested capture.
    """
    if isinstance(model.forward, PipelineGraphCache) or isinstance(model.encoder.forward, EncoderGraphCache):
        raise ValueError("disable existing graph caches before enabling pipeline graphs")
    cache = PipelineGraphCache(model, max_graphs)
    model.forward = cache
    return cache


def disable_pipeline_graphs(model):
    cache = model.forward
    if not isinstance(cache, PipelineGraphCache):
        return
    for device in {entry[1].device for entry in cache.entries.values()}:
        torch.cuda.synchronize(device)
    model.forward = cache.model_forward
    cache.entries.clear()
