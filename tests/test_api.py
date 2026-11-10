"""CPU contract tests of the public API and the CLI that need no GPU."""
from __future__ import annotations

import pytest
import torch

import kea_kernels
from conftest import write_fake_export
from kea_kernels import model as M
from kea_kernels.__main__ import main


def test_public_api():
    assert {"load", "transcribe", "close", "Model", "Transcript"} <= set(kea_kernels.__all__)


def test_load_refuses_unknown_modes_before_any_gpu_work(tmp_path):
    d = write_fake_export(tmp_path / "kea-t", "ternary")
    with pytest.raises(ValueError, match="supports modes"):
        kea_kernels.load(d, mode="x1")
    d = write_fake_export(tmp_path / "kea-b", "binary")
    with pytest.raises(ValueError, match="supports modes"):
        kea_kernels.load(d, mode="compact")


def test_device_check_refuses_cpu():
    with pytest.raises(ValueError, match="NVIDIA GPUs only"):
        M.check_device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) == (12, 0),
                    reason="needs a CUDA GPU that is not SM120")
def test_device_check_refuses_other_gpus():
    with pytest.raises(ValueError, match="SM120"):
        M.check_device("cuda")


def test_short_audio_is_padded_to_30_ms_and_arrays_must_be_mono():
    assert len(M._as_clip(torch.zeros(100))) == M.MIN_DECODE_SAMPLES
    assert len(M._as_clip([0.0] * 1000)) == 1000
    with pytest.raises(ValueError, match="1-D"):
        M._as_clip(torch.zeros(2, 100))


def test_read_audio_averages_channels_and_requires_16_khz(tmp_path):
    import numpy as np
    import soundfile
    stereo = np.stack([np.full(1600, 0.5, np.float32), np.full(1600, -0.25, np.float32)], axis=1)
    soundfile.write(tmp_path / "s.wav", stereo, 16000, subtype="FLOAT")
    assert np.allclose(M.read_audio(tmp_path / "s.wav"), 0.125)
    soundfile.write(tmp_path / "r.wav", stereo, 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="16 kHz"):
        M.read_audio(tmp_path / "r.wav")


def test_cli_reports_errors_with_exit_status_1(tmp_path, capsys):
    assert main([str(tmp_path), "clip.wav"]) == 1
    assert "manifest.json" in capsys.readouterr().err


def test_cache_dir_follows_the_environment(monkeypatch, tmp_path):
    from kea_kernels import _nvrtc
    monkeypatch.setenv("KEA_KERNELS_CACHE", str(tmp_path / "a"))
    assert _nvrtc.cache_dir() == tmp_path / "a"
    monkeypatch.delenv("KEA_KERNELS_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "x"))
    assert _nvrtc.cache_dir() == tmp_path / "x" / "kea-kernels"
