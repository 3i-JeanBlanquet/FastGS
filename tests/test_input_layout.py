import os
import pytest
from utils.input_layout import find_model_dir, resolve_image_path, resolve_depth_path

PREFIX = "/Users/3i-a1-2025-003/Documents/repositories/rnd.toolkit/pipeline"
SCENE_MESH = os.path.join(PREFIX, "rnd.project.mesh/test/input3")
RIG = "rig-1,0,0,0"

pytestmark = pytest.mark.skipif(not os.path.isdir(SCENE_MESH), reason="reference data absent")


def test_find_model_dir_accepts_bare_zero_dir():
    assert find_model_dir(SCENE_MESH) == os.path.join(SCENE_MESH, "0")


def test_resolve_image_path_preserves_rig_subdir():
    name = RIG + "/0179c0d4-f4d6-4003-b156-dbd7b2d3e8a9.jpg"
    got = resolve_image_path(SCENE_MESH, "images", name)
    assert got == os.path.join(SCENE_MESH, name)
    assert os.path.exists(got)


def test_resolve_image_path_distinguishes_the_four_faces():
    stem = "0179c0d4-f4d6-4003-b156-dbd7b2d3e8a9.jpg"
    rigs = [d for d in os.listdir(SCENE_MESH) if d.startswith("rig-")]
    paths = {resolve_image_path(SCENE_MESH, "images", r + "/" + stem) for r in rigs}
    assert len(paths) == 4, "the four cube faces must not collapse to one path"


def test_resolve_depth_path_finds_sibling_suffix_form():
    img = os.path.join(SCENE_MESH, RIG, "0179c0d4-f4d6-4003-b156-dbd7b2d3e8a9.jpg")
    assert resolve_depth_path(img) == img[:-4] + "_depth.png"


def test_resolve_depth_path_returns_none_when_absent(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"")
    assert resolve_depth_path(str(p)) is None
