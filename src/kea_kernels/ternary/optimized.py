"""Validated encoder and batch-one decoder fusions within NeMo."""
import contextlib
import torch

from .graphs import (EncoderGraphCache, PipelineGraphCache, enable_pipeline_graphs,
                     disable_encoder_graphs, disable_pipeline_graphs)
from .transforms import encoder_transform
from .decoder_kernels import decoder_transform, control_transform
from .graph_memory import decoder_graph_pool


@torch.no_grad()
def enable_optimizations(model, *, max_graphs=4, decoder=True, position_dot=True, encoder_storage="packed"):
    """Enable residual INT8x3, fused operations, and pipeline graphs.

    This is an inference-only transformation of the existing packed model.
    Disable before editing weights, moving devices, saving state, or training.
    Decoder fusions preserve NeMo's greedy TDT algorithm and floating weights.
    Set decoder=False, position_dot=False for the second-pass arithmetic.
    Conditional decoder graph allocations are protected in either mode.
    encoder_storage="expanded" adds an exact INT8 column-major weight cache,
    tuned 5090 tiles and Conformer fusions, trading roughly 604 MB for speed.
    Calls must run without gradients
    or autocast. Only offline relative self-attention is supported.
    """
    if encoder_storage not in ("packed", "expanded"):
        raise ValueError("encoder_storage must be packed or expanded")
    if model.training or hasattr(model, "_packed_optimization"):
        raise ValueError("expected an eval model without existing optimizations")
    if isinstance(model.forward, PipelineGraphCache) or isinstance(model.encoder.forward, EncoderGraphCache):
        raise ValueError("disable existing graph caches before enabling optimizations")
    from .runtime import PackedLinear
    if not all(isinstance(layer.self_attn.linear_q, PackedLinear) and layer.self_attention_model == "rel_pos"
               for layer in model.encoder.layers):
        raise ValueError("expected the packed relative-position Parakeet encoder")
    ctx = contextlib.ExitStack()
    try:
        projection = None
        if encoder_storage == "expanded":
            from .encoder4 import ExpandedProjection
            projection = ExpandedProjection("triton_col_tuned")
        ctx.enter_context(decoder_graph_pool(model))
        ctx.enter_context(encoder_transform(model, mode="int8x3", qkv=True, silu=True, position=True,
                          fold_bn=True, norm=True, attention=True, norm_quant=True, position_dot=position_dot,
                          projection=projection))
        if encoder_storage == "expanded":
            from .layer_fusions4 import layer_fusions
            ctx.enter_context(layer_fusions(model, projection))
        if decoder:
            ctx.enter_context(decoder_transform(model,joint=True,lstm=True,precompute=True,block=1))
            ctx.enter_context(control_transform(model,storage=True))
    except BaseException:
        ctx.close()
        raise
    original = model.encoder.forward
    device = next(model.parameters()).device
    def forward(audio_signal, length, **kwargs):
        if (model.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or audio_signal.device != device or audio_signal.dtype != torch.float32
                or any(v is not None and v is not False for v in kwargs.values())):
            raise RuntimeError("optimized encoder requires offline FP32 inference on its original device; disable optimizations first")
        return original(audio_signal=audio_signal, length=length)
    model.encoder.forward = forward
    model._packed_optimization = (ctx, original)
    try:
        if max_graphs:
            enable_pipeline_graphs(model, max_graphs=max_graphs)
    except BaseException:
        disable_optimizations(model)
        raise
    return model


@torch.no_grad()
def disable_optimizations(model):
    """Synchronize, release captured graphs, and restore the original modules."""
    state = getattr(model, "_packed_optimization", None)
    if state is None:
        return
    disable_pipeline_graphs(model)
    disable_encoder_graphs(model)
    ctx, original = state
    model.encoder.forward = original
    ctx.__exit__(None, None, None)
    del model._packed_optimization
