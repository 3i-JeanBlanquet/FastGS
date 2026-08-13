import os
import pytest
from scripts.estimate_depth_scale import estimate_scale

PREFIX = "/Users/3i-a1-2025-003/Documents/repositories/rnd.toolkit/pipeline"
SCENE_MESH = os.path.join(PREFIX, "rnd.project.mesh/test/input3")
SCENE_RECON = os.path.join(PREFIX, "rnd.project.reconstruction/test/dest")

pytestmark = pytest.mark.skipif(not os.path.isdir(SCENE_MESH), reason="reference data absent")


def test_mesh_scene_scale_and_convention():
    r = estimate_scale(SCENE_MESH)
    assert r["convention"] == "planar"
    assert 3900 < r["scale_mm_per_unit"] < 4400
    assert r["n_samples"] > 10000
    assert r["convention_margin"] > 1.2


def test_scale_is_per_scene_not_a_constant():
    # The two captures differ by ~5%. A hardcoded constant is therefore wrong.
    a = estimate_scale(SCENE_MESH)["scale_mm_per_unit"]
    b = estimate_scale(SCENE_RECON, images_arg="../source/images")["scale_mm_per_unit"]
    assert abs(a - b) / b > 0.02, "scenes must be measurably different"


def test_outliers_are_reported():
    r = estimate_scale(SCENE_MESH)
    assert isinstance(r["outliers"], list)
    assert len(r["outliers"]) < 0.5 * len(r["per_image"])
