"""CPU contract tests: weight code layouts, export checks and model-directory detection."""
from __future__ import annotations

import json

import pytest
import torch

from conftest import write_fake_export
from kea_kernels import formats
from kea_kernels.binary import bits


def test_binary_codes_round_trip_and_layout():
    signs = torch.where(torch.rand(5, 21) < 0.5, -1, 1).to(torch.int8)
    packed = bits.pack_signs(signs)
    assert packed.dtype == torch.uint8 and packed.shape == (5, 3)
    assert torch.equal(bits.unpack_signs(packed, 21), signs)
    # code j of a row is bit j % 8 of byte j // 8, LSB first; 1 = +1
    one = torch.full((1, 8), -1, dtype=torch.int8)
    one[0, 3] = 1
    assert bits.pack_signs(one).tolist() == [[1 << 3]]


def test_binary_codes_reject_tail_bits_and_bad_scales():
    codes = torch.zeros(2, 2, dtype=torch.uint8)
    codes[0, 1] = 0x80  # bit 15 set for K = 12: tail padding
    with pytest.raises(ValueError, match="tail"):
        bits.unpack_bits(codes, 12)
    with pytest.raises(ValueError, match="tail"):
        bits.validate_weights(codes, torch.ones(2), None, 12)
    good = torch.zeros(2, 2, dtype=torch.uint8)
    assert bits.validate_weights(good, torch.ones(2), None, 12) == (2, 12)
    with pytest.raises(ValueError, match="finite"):
        bits.validate_weights(good, torch.tensor([1.0, float("nan")]), None, 12)
    with pytest.raises(ValueError, match="codes must be"):
        bits.validate_weights(good, torch.ones(2), None, 20)


def test_dense_weight_is_sign_times_row_scale():
    signs = torch.tensor([[1, -1, 1], [-1, -1, 1]], dtype=torch.int8)
    w = bits.dense_weight(bits.pack_signs(signs), torch.tensor([0.5, 2.0]), 3)
    assert torch.equal(w, torch.tensor([[0.5, -0.5, 0.5], [-2.0, -2.0, 2.0]], dtype=torch.float64))


def test_ternary_codes_round_trip_and_layout():
    codes = torch.randint(-1, 2, (4, 13), dtype=torch.int8)
    packed = formats.pack_ternary(codes)
    assert packed.shape == (4, 4)
    assert torch.equal(formats.unpack_ternary(packed, 13), codes)
    # 0 -> 00, +1 -> 01, -1 -> 10, code j at bits 2 * (j % 4)
    assert formats.pack_ternary(torch.tensor([[0, 1, -1, 1]], dtype=torch.int8)).tolist() == [[0b01_10_01_00]]


def test_ternary_codes_reject_the_unused_pattern():
    with pytest.raises(ValueError):
        formats.unpack_ternary(torch.tensor([[0b11]], dtype=torch.uint8), 4)
    with pytest.raises(ValueError):
        formats.unpack_ternary(torch.tensor([[0b01_00_00_00]], dtype=torch.uint8), 3)  # tail code set


@pytest.mark.parametrize("kind", ["binary", "ternary"])
def test_check_export_accepts_an_intact_export(tmp_path, kind):
    d = write_fake_export(tmp_path / "m", kind)
    manifest = formats.check_export(d, kind)
    assert len(manifest["quantized_layers"]) == formats.EXPECTED_MODULES


def test_check_export_refuses_the_other_format(tmp_path):
    d = write_fake_export(tmp_path / "m", "ternary")
    with pytest.raises(ValueError, match="expected 'parakeet-binary-v1'"):
        formats.check_export(d, "binary")


def test_check_export_refuses_a_modified_file(tmp_path):
    d = write_fake_export(tmp_path / "m", "binary")
    (d / "tokenizer" / "vocab.txt").write_text("b\n")
    with pytest.raises(ValueError, match="SHA-256"):
        formats.check_export(d, "binary")
    formats.check_export(d, "binary", verify_hashes=False)


def test_check_export_refuses_missing_layers(tmp_path):
    d = write_fake_export(tmp_path / "m", "binary")
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["quantized_layers"].pop(next(iter(manifest["quantized_layers"])))
    (d / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="264"):
        formats.check_export(d, "binary")


def test_check_export_refuses_wrong_code_shapes(tmp_path):
    d = write_fake_export(tmp_path / "m", "binary", n=8, k=16)
    manifest = json.loads((d / "manifest.json").read_text())
    for info in manifest["quantized_layers"].values():
        info["shape"] = [8, 32]
    (d / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="codes must be uint8"):
        formats.check_export(d, "binary", verify_hashes=False)


def test_describe_recognises_the_three_downloads(tmp_path):
    keab = write_fake_export(tmp_path / "kea-b", "binary")
    keat = write_fake_export(tmp_path / "kea-t", "ternary")
    root = tmp_path / "kea-b-0.1"
    write_fake_export(root / "compat", "binary", extra={"keab": {"bundle_sha256": "abc"}})
    (root / "manifest.json").write_text(json.dumps({"format": formats.BUNDLE_FORMAT, "sha256": "abc"}))
    assert (formats.describe(keab).name, formats.describe(keab).default_mode) == ("kea-b", "x2")
    assert (formats.describe(keat).name, formats.describe(keat).default_mode) == ("kea-t", "x2")
    info = formats.describe(root)
    assert (info.name, info.kind, info.default_mode, info.export) == ("kea-b-0.1", "binary", "x1",
                                                                       (root / "compat").resolve())
    assert formats.describe(root / "compat").default_mode == "x1"


def test_describe_refuses_a_bundle_without_its_export_or_mismatched(tmp_path):
    root = tmp_path / "kea-b-0.1"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"format": formats.BUNDLE_FORMAT, "sha256": "abc"}))
    with pytest.raises(ValueError, match="compat"):
        formats.describe(root)
    write_fake_export(root / "compat", "binary", extra={"keab": {"bundle_sha256": "other"}})
    with pytest.raises(ValueError, match="bundle"):
        formats.describe(root)


def test_describe_refuses_unknown_directories(tmp_path):
    with pytest.raises(ValueError, match="manifest.json"):
        formats.describe(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError, match="unsupported format"):
        formats.describe(tmp_path)
