"""GPU contract tests (SM120 only): kernel outputs against dense references computed independently on the host."""
from __future__ import annotations

import types

import pytest
import torch

from kea_kernels.binary import bits, configs, layer as L, layer_parts

pytestmark = pytest.mark.gpu

SHAPES = [(1024, 1024), (4096, 1024), (1024, 4096), (3072, 1024)]


def _weights(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    signs = torch.where(torch.rand(n, k, generator=g) < 0.5, -1, 1).to(torch.int8)
    scale = torch.rand(n, generator=g) * 0.02 + 0.001
    return bits.pack_signs(signs), scale.float(), signs


def _exact_epilogue(q, qs, signs, scale, parts):
    """FP64 sum_c (q_c @ signs^T) * s_c * alpha from exact integer accumulators."""
    acc = [q[c].cpu().to(torch.int64) @ signs.to(torch.int64).t() for c in range(parts)]
    y = sum(a.to(torch.float64) * qs[c].cpu().to(torch.float64)[:, None] for c, a in enumerate(acc))
    return y * scale.to(torch.float64)[None, :]


@pytest.mark.parametrize("parts", [1, 2, 3])
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [38, 126, 300])
@torch.inference_mode()
def test_b1_dot_equals_exact_integer_reference(parts, n, k, m):
    codes, scale, signs = _weights(n, k, seed=n + k + m)
    layer = layer_parts.LAYERS[parts].prepare(codes, scale, None, k, "cuda")
    assert layer.fast
    x = torch.randn(m, k, device="cuda")
    q, qs, ldq = L.quantize(x, (0, x.stride(0), x.stride(1), m), m, k)
    y = torch.empty(m, n, device="cuda")
    layer.dot(q[:parts], qs[:parts], ldq, m, y, parts=parts)
    torch.cuda.synchronize()
    ref = _exact_epilogue(q[:, :, :k], qs, signs, scale, parts)
    torch.testing.assert_close(y.cpu().double(), ref, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("parts,tol", [(1, 2e-2), (2, 1e-4), (3, 1e-6)])
@torch.inference_mode()
def test_linear_matches_dense_fp64_within_activation_precision(parts, tol):
    n, k, m = 1024, 4096, 126
    codes, scale, signs = _weights(n, k, seed=7)
    layer = layer_parts.LAYERS[parts].prepare(codes, scale, None, k, "cuda")
    x = torch.randn(2, m // 2, k, device="cuda")
    y = layer.linear(x).cpu().double()
    ref = x.cpu().double() @ bits.dense_weight(codes, scale, k).t()
    err = (y - ref).norm() / ref.norm()
    assert err < tol


@pytest.mark.parametrize("parts", [2, 3])
@pytest.mark.parametrize("m,n,k", [(38, 1024, 1024), (38, 1024, 4096), (126, 1024, 4096)])
@torch.inference_mode()
def test_exact_split_k_finish_equals_in_kernel_finish(parts, m, n, k):
    """The deferred XCHG 5 integer slices, finished by the row epilogue, give bitwise the in-kernel finish."""
    from kea_kernels.binary import hybrid_exact, hybrid_x2, row_exact, row_exact_x2
    from kea_kernels.ternary import row_fusion
    row_exact_x2.install()
    codes, scale, _ = _weights(n, k, seed=m * n + k)
    layer = (layer_parts.LAYERS[2] if parts == 2 else L.Int8MmaLayer).prepare(codes, scale, None, k, "cuda")
    cfg = hybrid_exact.choose_exact_deferred(m, n, k)
    assert cfg is not None and cfg.cluster > 1
    xx = torch.randn(1, m, k, device="cuda")
    q, qs = layer_parts.m1_quantize_parts(xx, parts)
    provider = types.SimpleNamespace(row_warps="adaptive", prequantized={},
                                     null=torch.empty(0, device="cuda", dtype=torch.uint8))
    residual = torch.randn(1, m, n, device="cuda")
    slices = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
    if parts == 2:
        hybrid_x2.dot_slices_x2(layer, q, qs, k, m, slices, cfg)
        value = row_exact_x2.ExactDeferredProjectionX2(slices, layer.scale, layer.bias, cfg.cluster, (1, m, n), qs, 0, 2)
    else:
        hybrid_exact.dot_slices(layer, q, qs, k, m, slices, cfg)
        value = row_exact.ExactDeferredProjection(slices, layer.scale, layer.bias, cfg.cluster, (1, m, n), qs, 0)
    exact = row_fusion.finish_residual(provider, value, residual, 1.0)
    y = torch.empty(1, m, n, device="cuda")
    layer.dot(q, qs, k, m, y.view(m, n), config=configs.choose(m, n, k), **({"parts": 2} if parts == 2 else {}))
    inkernel = row_fusion.finish_residual(provider, row_fusion.DeferredProjection(y, layer.scale, layer.bias, 1,
                                                                                  (1, m, n)), residual, 1.0)
    torch.cuda.synchronize()
    assert torch.equal(exact, inkernel)


@torch.inference_mode()
def test_ternary_packed_linear_matches_dense_fp64():
    from kea_kernels import formats
    from kea_kernels.ternary.runtime import PackedLinear
    n, k, m = 1024, 1024, 64
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(-1, 2, (n, k), generator=g, dtype=torch.int8)
    scale = torch.rand(n, generator=g) * 0.02 + 0.001
    module = PackedLinear(formats.pack_ternary(codes), scale.float(), k).cuda()
    x = torch.randn(m, k, device="cuda")
    y = module(x).cpu().double()
    ref = x.cpu().double() @ (codes.double() * scale.double()[:, None]).t()
    assert (y - ref).norm() / ref.norm() < 1e-5
