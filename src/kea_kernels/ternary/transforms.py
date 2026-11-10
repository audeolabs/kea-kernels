"""Reversible encoder transforms: fused QKV, SiLU, positions, folded batch norm, LayerNorm, attention, INT8x3."""
import contextlib
import functools
import types
import torch
from .kernels import matmul
from .runtime import PackedLinear


@contextlib.contextmanager
def encoder_transform(model, *, mode="bf16x3", qkv=False, silu=False, position=False,
               sdpa=False, fold_bn=False, config=None, norm=False, attention=False, atomic=False,
               norm_quant=False, position_dot=False, projection=None):
    """Apply reversible eval-only changes; exit only after GPU work completes.

    Positional projections are frozen only while capturing an offline encoder
    graph, after an eager warmup on the identical shape. Keep this context alive
    for the entire lifetime of graphs captured inside it.
    """
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("experiments require eval and no gradients")
    undo, keepalive = [], []
    norms = {}
    from .fused_kernels import int8_matmul
    project_integer = projection or int8_matmul
    if norm_quant:
        if not mode.startswith("int8") or not qkv:
            raise ValueError("fused normalization requires integer arithmetic and fused QKV")
        for layer in model.encoder.layers:
            norms.update({id(layer.feed_forward1.linear1): layer.norm_feed_forward1,
                          id(layer.feed_forward2.linear1): layer.norm_feed_forward2,
                          id(layer.conv.pointwise_conv1): layer.norm_conv,
                          id(layer.self_attn.linear_q): layer.norm_self_att})

    def patch(obj, key, value):
        # Preserve inherited methods without leaving bound methods in __dict__.
        existed, old = key in obj.__dict__, getattr(obj, key, None)
        setattr(obj, key, value)
        undo.append((obj, key, existed, old))

    try:
        for mod in model.modules():
            if isinstance(mod, PackedLinear):
                patch(mod, "mode", mode)
                if mode.startswith("int8"):
                    from .runtime import PackedPointwiseConv1d
                    components = {"int8": 1, "int8x2": 2, "int8x3": 3}[mode]
                    patch(mod, "forward", lambda x, mod=mod: project_integer(x, mod, isinstance(mod, PackedPointwiseConv1d), components,
                                                                         norm=norms.get(id(mod)), config=config))
                elif config is not None or atomic:
                    from .runtime import PackedPointwiseConv1d
                    def run(x, mod=mod):
                        return matmul(x, mod.packed_t, mod.scale, mod.bias, mod.in_features,
                                      mode=mod.mode, conv=isinstance(mod, PackedPointwiseConv1d), config=config, atomic=atomic)
                    patch(mod, "forward", run)
            elif norm and isinstance(mod, torch.nn.LayerNorm):
                from .fused_kernels import layer_norm
                patch(mod, "forward", lambda x, mod=mod: layer_norm(x, mod))
        for layer in model.encoder.layers:
            att = layer.self_attn
            if norm_quant:
                for ln in (layer.norm_feed_forward1, layer.norm_feed_forward2, layer.norm_conv, layer.norm_self_att):
                    patch(ln, "forward", lambda x: x)
            if attention:
                from .fused_kernels import relative_attention
                patch(att, "forward", types.MethodType(functools.partial(relative_attention, position_dot=position_dot), att))
            if sdpa:
                patch(att, "use_pytorch_sdpa", True)
            if qkv:
                mods = [att.linear_q, att.linear_k, att.linear_v]
                weights = torch.cat([m.packed_t for m in mods], dim=1).contiguous()
                scales = torch.cat([m.scale for m in mods])
                biases = torch.cat([m.bias for m in mods]) if mods[0].bias is not None else None
                keepalive.extend([weights, scales, biases])
                original = att.forward_qkv
                def forward(query, key, value, att=att, weights=weights, scales=scales,
                            biases=biases, original=original):
                    if query is not key or query is not value:
                        return original(query, key, value)
                    if mode.startswith("int8"):
                        mod = types.SimpleNamespace(packed_t=weights, scale=scales, bias=biases, out_features=scales.numel())
                        y = project_integer(query, mod, components=components, norm=norms.get(id(att.linear_q)), config=config)
                    else:
                        y = matmul(query, weights, scales, biases, att.h * att.d_k, mode=mode)
                    return tuple(v.reshape(query.shape[0], -1, att.h, att.d_k).transpose(1, 2)
                                 for v in y.chunk(3, dim=-1))
                patch(att, "forward_qkv", forward)
            if silu:
                for ff in (layer.feed_forward1, layer.feed_forward2):
                    def ff_forward(x, ff=ff):
                        m = ff.linear1
                        if mode.startswith("int8"):
                            y = project_integer(x, m, components=components, activation="silu", norm=norms.get(id(m)), config=config)
                        else:
                            y = matmul(x, m.packed_t, m.scale, m.bias, m.in_features,
                                       mode=m.mode, activation="silu", config=config)
                        return ff.linear2(y)
                    patch(ff, "forward", ff_forward)
            if position:
                proj = att.linear_pos
                original = proj.forward
                cache = []
                captured = []
                patch(proj, "_packed_positional_capture", captured)
                keepalive.append(cache)
                keepalive.append(captured)
                def positional(x, original=original, cache=cache, captured=captured):
                    if torch.cuda.is_current_stream_capturing():
                        if not cache:
                            raise RuntimeError("warm positional projection before capture")
                        captured.append(cache[0])
                        return cache[0]
                    result = original(x)
                    cache[:] = [result]
                    return result
                patch(proj, "forward", positional)
            if fold_bn:
                conv = layer.conv
                if not isinstance(conv.batch_norm, torch.nn.BatchNorm1d):
                    raise ValueError("batch-norm folding requires BatchNorm1d")
                bn, dw = conv.batch_norm, conv.depthwise_conv
                alpha = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                weight = dw.weight.detach().clone()
                bias = dw.bias
                dw.weight.copy_(weight * alpha[:, None, None])
                dw.bias = torch.nn.Parameter(((bias if bias is not None else 0) - bn.running_mean) * alpha + bn.bias,
                                             requires_grad=False)
                keepalive.append((dw, weight, bias))
                patch(bn, "forward", lambda x: x)
        yield keepalive
    finally:
        # Also safe when the context is finalized after its caller leaves
        # inference_mode: model tensors may have been created inside that mode.
        with torch.inference_mode():
            torch.cuda.synchronize()
            for item in keepalive:
                if isinstance(item, tuple) and len(item) == 3:
                    dw, weight, bias = item
                    dw.weight.copy_(weight)
                    dw.bias = bias
            for obj, key, existed, old in reversed(undo):
                if existed:
                    setattr(obj, key, old)
                else:
                    delattr(obj, key)
