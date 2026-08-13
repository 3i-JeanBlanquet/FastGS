import os
import numpy as np
import torch
import pytest
from PIL import Image

from utils.depth_utils import load_depth


def _write_depth_png(path, arr):
    """Write a uint16 millimetre depth PNG with known values."""
    Image.fromarray(arr.astype(np.uint16), mode="I;16").save(path)


# Synthetic tests (always run)

def test_shape_and_dtype(tmp_path):
    arr = np.full((8, 8), 2000, dtype=np.uint16)  # 2 m everywhere
    path = str(tmp_path / "depth.png")
    _write_depth_png(path, arr)

    d, m = load_depth(path, 1000.0, (8, 8), 30.0)

    assert d.shape == (1, 8, 8) and d.dtype == torch.float32
    assert m.shape == (1, 8, 8) and m.dtype == torch.bool


def test_zero_pixels_are_masked_out(tmp_path):
    arr = np.full((4, 4), 2000, dtype=np.uint16)
    arr[0, 0] = 0
    arr[1, 1] = 0
    path = str(tmp_path / "depth.png")
    _write_depth_png(path, arr)

    d, m = load_depth(path, 1000.0, (4, 4), 30.0)

    assert not m[0, 0, 0]
    assert not m[0, 1, 1]
    assert m.sum().item() == 14  # 16 - 2 zeroed pixels
    assert not torch.isnan(d).any()


def test_values_converted_to_colmap_units_not_millimetres(tmp_path):
    # 4135.3 mm/unit, pixel value 7000 mm -> ~1.693 COLMAP units, not 7000.
    arr = np.full((2, 2), 7000, dtype=np.uint16)
    path = str(tmp_path / "depth.png")
    _write_depth_png(path, arr)

    scale = 4135.3
    d, m = load_depth(path, scale, (2, 2), 30.0)

    expected = 7000.0 / scale
    assert m.all()
    assert torch.allclose(d, torch.full((1, 2, 2), expected), atol=1e-4)
    assert d.max().item() < 10.0  # sanity: nowhere near millimetre-scale


def test_depth_max_rejects_far_pixels(tmp_path):
    arr = np.full((4, 4), 20000, dtype=np.uint16)  # 20 m everywhere
    path = str(tmp_path / "depth.png")
    _write_depth_png(path, arr)

    scale = 1000.0
    _, m_far = load_depth(path, scale, (4, 4), 30.0)   # 30 m cutoff: keeps everything
    _, m_near = load_depth(path, scale, (4, 4), 3.0)   # 3 m cutoff: rejects everything

    assert m_far.sum().item() == 16
    assert m_near.sum().item() == 0
    assert m_near.sum() < m_far.sum()


def test_resize_matches_requested_resolution(tmp_path):
    arr = np.full((8, 8), 3000, dtype=np.uint16)
    path = str(tmp_path / "depth.png")
    _write_depth_png(path, arr)

    d, m = load_depth(path, 1000.0, (4, 4), 30.0)

    assert d.shape == (1, 4, 4)
    assert m.shape == (1, 4, 4)
    assert m.all()


def test_all_invalid_depth_produces_empty_mask_without_crashing(tmp_path):
    arr = np.zeros((4, 4), dtype=np.uint16)  # every pixel invalid
    path = str(tmp_path / "depth.png")
    _write_depth_png(path, arr)

    d, m = load_depth(path, 1000.0, (4, 4), 30.0)

    assert m.sum().item() == 0
    assert not torch.isnan(d).any()
    assert torch.all(d == 0)


# Real-capture test (skipped when the reference data is absent)

PREFIX = "/Users/3i-a1-2025-003/Documents/repositories/rnd.toolkit/pipeline"
REAL_DEPTH = os.path.join(PREFIX, "rnd.project.mesh/test/input3",
                           "rig-1,0,0,0/0179c0d4-f4d6-4003-b156-dbd7b2d3e8a9_depth.png")
REAL_SCALE = 4135.3


@pytest.mark.skipif(not os.path.exists(REAL_DEPTH), reason="reference data absent")
def test_real_capture_median_lands_in_colmap_units():
    d, m = load_depth(REAL_DEPTH, REAL_SCALE, (1024, 1024), 30.0)
    assert m.sum() > 0
    med = d[m].median().item()
    assert 0.2 < med < 8.0, "expected COLMAP units, got {}".format(med)
