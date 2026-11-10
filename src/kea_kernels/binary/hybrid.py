"""The hybrid provider protocol and pipeline for the 1-bit layers.

The pipeline is the ternary runtime's fastest batch-one stack: encoder_transform (int8x3, fused QKV, SiLU, position,
folded batch norm, LayerNorm, attention, fused input quantization, position dot) + layer_fusions(residual=False) +
row_fusions + the fused decoder and the pipeline graph cache. Its provider (a) returns DeferredProjection objects for
the N = 1024, no-activation projections, so row_fusion's _row_epilogue folds the split-K finish, the residual add, the
next LayerNorm and the next INT8x3 quantization into one kernel, and (b) consumes those prequantized activations
(provider.prequantized) instead of re-quantizing.

`HybridProjection` implements that provider protocol (defer_epilogue, prequantized, no_defer_weights, row_warps, null)
over the 1-bit layer table of provider.load_binary_model; the deployed providers are its subclasses
(hybrid_exact.HybridProjectionExact at x3, hybrid_x2.HybridProjectionX2 at x2). `hybrid_pipeline` applies the
context stack.
"""
from __future__ import annotations

import contextlib

import torch


class HybridProjection:
    """Provider state shared by the hybrid providers (the subclasses implement __call__)."""

    def __init__(self, table, names=None):
        from kea_kernels.binary.provider import BinaryProjection
        self._resolver = BinaryProjection(table, names)  # handle -> (entry, offset, count) lookup only
        self.table = table
        self.defer_epilogue = False
        self.prequantized: dict = {}
        self.no_defer_weights: set = set()
        self.row_warps = "adaptive"
        self.null = torch.empty(0, device="cuda", dtype=torch.uint8)
        if self.null.data_ptr() != 0:
            raise RuntimeError("expected a null empty-tensor pointer")
        self.calls = {"deferred": 0, "prequantized": 0, "quantized": 0}
        # Deferred N = 1024 calls may use split-K slices (the subclasses use exact integer slices); False keeps
        # b1_dot's exact in-kernel finish (SPLIT = 1).
        self.split_deferred = True

    @property
    def bytes(self) -> int:
        return self.table.weight_bytes()


@contextlib.contextmanager
def hybrid_pipeline(model, provider):
    """The hybrid context stack with `provider`: encoder_transform / layer_fusions(residual=False) / row_fusions
    inside decoder_graph_pool, then the decoder and control transforms, then the pipeline graph cache."""
    from kea_kernels.ternary.decoder_kernels import control_transform, decoder_transform
    from kea_kernels.ternary.graph_memory import decoder_graph_pool
    from kea_kernels.ternary.graphs import disable_pipeline_graphs, enable_pipeline_graphs
    from kea_kernels.ternary.layer_fusions4 import layer_fusions
    from kea_kernels.ternary.row_fusion import row_fusions
    from kea_kernels.ternary.transforms import encoder_transform
    with decoder_graph_pool(model), \
            encoder_transform(model, mode="int8x3", qkv=True, silu=True, position=True, fold_bn=True, norm=True,
                              attention=True, norm_quant=True, position_dot=True, projection=provider), \
            layer_fusions(model, provider, residual=False), row_fusions(model, provider), \
            decoder_transform(model, joint=True, lstm=True, precompute=True, block=1), \
            control_transform(model, storage=True):
        enable_pipeline_graphs(model, max_graphs=4)
        try:
            yield provider
        finally:
            disable_pipeline_graphs(model)
