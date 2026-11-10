"""CPU contract tests of the kernel builders: weight fragment layout, part-count sources and the provider adapter."""
from __future__ import annotations

import pytest
import torch

from kea_kernels.binary import configs, kernels, kernels_exact, kernels_exact_x2, kernels_parts
from kea_kernels.binary.hybrid_exact import TUNED_EXACT_DEFERRED
from kea_kernels.binary.provider import PartCount


@pytest.mark.parametrize("n,k", [(8, 128), (24, 1024), (5, 300)])
def test_fragment_layout_round_trips(n, k):
    bits = torch.randint(0, 2, (n, k), dtype=torch.uint8)
    words = kernels.relayout(bits)
    assert words.shape == (-(-n // 8), -(-k // 128), 32) and words.dtype == torch.int32
    assert torch.equal(kernels.unlayout(words, n, k), bits)


def test_every_parakeet_shape_has_tuned_tiles():
    for n, k in [(1024, 1024), (1024, 4096), (2048, 1024), (3072, 1024), (4096, 1024)]:
        for m in (1, 38, 126, 400, 1500):
            assert configs.choose(m, n, k) in configs.TUNED_M.values()


@pytest.mark.parametrize("parts", [1, 2, 3])
def test_part_count_sources_cover_exactly_p_parts(parts):
    cfg = configs.choose(126, 1024, 1024)
    source, name = kernels_parts.dot_source(kernels_parts.PartsConfig.of(cfg, parts), False, 0, parts)
    assert name.startswith(f"x{parts}_b1_dot_")
    assert f"constexpr int PARTS = {parts};" in source
    assert "for (int c = 0; c < PARTS; ++c)" in source
    with pytest.raises(ValueError):
        kernels_parts.transform(kernels.dot_source(cfg, False)[0], 4)


def test_part_count_shrinks_the_activation_tile():
    cfg = configs.choose(126, 1024, 1024)
    sizes = [kernels_parts.PartsConfig.of(cfg, p).shared_bytes for p in (1, 2, 3)]
    assert sizes[0] < sizes[1] < sizes[2] == cfg.shared_bytes


def test_exact_deferred_sources():
    for cfg in filter(None, TUNED_EXACT_DEFERRED.values()):
        s3, n3 = kernels_exact.dot_source(cfg)
        s2, n2 = kernels_exact_x2.dot_source(cfg, 0, 2)
        assert "XCHG == 5" in s3 and n2 == "x2_" + n3 and "constexpr int PARTS = 2;" in s2
        assert kernels_exact_x2.dot_source(cfg, 0, 3) == (s3, n3)


def test_part_count_overrides_components_and_forwards_writes():
    calls = []

    class Provider:
        defer_epilogue = False

        def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None,
                     input_activation=""):
            calls.append(components)
            return x

    inner = Provider()
    wrapped = PartCount(inner, 2)
    wrapped(torch.zeros(1), None, components=3)
    assert calls == [2]
    wrapped.defer_epilogue = True
    assert inner.defer_epilogue is True and wrapped.n == 2
    with pytest.raises(ValueError):
        PartCount(inner, 4)
