"""End-to-end GPU contract tests on the real models (SM120 only). They need downloaded models and a speech clip:

    KEA_KERNELS_TEST_MODELS=DIR   containing kea-b-0.1/, kea-b/ and/or kea-t/ (hf download ... --local-dir DIR/NAME)
    KEA_KERNELS_TEST_AUDIO=FILE   a 16 kHz English speech clip
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import kea_kernels

pytestmark = pytest.mark.gpu
MODELS = Path(os.environ.get("KEA_KERNELS_TEST_MODELS", "/nonexistent"))
AUDIO = os.environ.get("KEA_KERNELS_TEST_AUDIO")
CASES = [("kea-b-0.1", None, "x1"), ("kea-b", None, "x2"), ("kea-b", "x3", "x3"), ("kea-t", None, "x2"),
         ("kea-t", "x3", "x3"), ("kea-t", "compact", "compact")]


@pytest.mark.parametrize("name,mode,expected_mode", CASES)
def test_load_transcribe_close(name, mode, expected_mode):
    if not (MODELS / name).is_dir() or not AUDIO:
        pytest.skip("set KEA_KERNELS_TEST_MODELS and KEA_KERNELS_TEST_AUDIO")
    with kea_kernels.load(MODELS / name, mode=mode) as model:
        assert (model.name, model.mode) == (name, expected_mode)
        text = kea_kernels.transcribe(model, AUDIO)
        assert isinstance(text, str) and text.strip()
        again = kea_kernels.transcribe(model, [AUDIO, AUDIO], return_tokens=True)
        assert [r.text for r in again] == [text, text]  # deterministic, graph-cached replay
        assert again[0].tokens and len(again[0].tokens) == len(again[0].timestamps)
        assert kea_kernels.decoder_graphs_mode(model) == "full_graph" or mode == "compact"
        silence = kea_kernels.transcribe(model, torch.zeros(100))  # shorter than 30 ms: padded, still decodes
        assert isinstance(silence, str)
    assert model.closed
    model.close()  # idempotent
    with pytest.raises(ValueError, match="closed"):
        kea_kernels.transcribe(model, AUDIO)
    torch.cuda.synchronize()
