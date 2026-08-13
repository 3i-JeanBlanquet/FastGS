import os
import pytest
from utils.input_layout import find_model_dir, resolve_image_path, resolve_depth_path, detect_scene_type

PREFIX = "/Users/3i-a1-2025-003/Documents/repositories/rnd.toolkit/pipeline"
SCENE_MESH = os.path.join(PREFIX, "rnd.project.mesh/test/input3")
RIG = "rig-1,0,0,0"

_HAS_REAL_DATA = os.path.isdir(SCENE_MESH)


# Tests using real capture data (may be skipped)

@pytest.mark.skipif(not _HAS_REAL_DATA, reason="reference data absent")
def test_find_model_dir_accepts_bare_zero_dir():
    assert find_model_dir(SCENE_MESH) == os.path.join(SCENE_MESH, "0")


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="reference data absent")
def test_resolve_image_path_preserves_rig_subdir():
    name = RIG + "/0179c0d4-f4d6-4003-b156-dbd7b2d3e8a9.jpg"
    got = resolve_image_path(SCENE_MESH, "images", name)
    assert got == os.path.join(SCENE_MESH, name)
    assert os.path.exists(got)


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="reference data absent")
def test_resolve_image_path_distinguishes_the_four_faces():
    stem = "0179c0d4-f4d6-4003-b156-dbd7b2d3e8a9.jpg"
    rigs = [d for d in os.listdir(SCENE_MESH) if d.startswith("rig-")]
    paths = {resolve_image_path(SCENE_MESH, "images", r + "/" + stem) for r in rigs}
    assert len(paths) == 4, "the four cube faces must not collapse to one path"


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="reference data absent")
def test_resolve_depth_path_finds_sibling_suffix_form():
    img = os.path.join(SCENE_MESH, RIG, "0179c0d4-f4d6-4003-b156-dbd7b2d3e8a9.jpg")
    assert resolve_depth_path(img) == img[:-4] + "_depth.png"


# Synthetic tests (always run)

def test_find_model_dir_with_sparse_zero(tmp_path):
    """Test find_model_dir locating sparse/0."""
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.bin").write_bytes(b"")
    assert find_model_dir(str(tmp_path)) == str(sparse_dir)


def test_find_model_dir_with_sparse_only(tmp_path):
    """Test find_model_dir locating sparse without /0."""
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    (sparse_dir / "cameras.bin").write_bytes(b"")
    assert find_model_dir(str(tmp_path)) == str(sparse_dir)


def test_find_model_dir_with_bare_zero(tmp_path):
    """Test find_model_dir locating bare 0 directory."""
    zero_dir = tmp_path / "0"
    zero_dir.mkdir()
    (zero_dir / "cameras.bin").write_bytes(b"")
    assert find_model_dir(str(tmp_path)) == str(zero_dir)


def test_resolve_image_path_flat_layout(tmp_path):
    """Test resolve_image_path with flat images directory."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img_file = images_dir / "test.jpg"
    img_file.write_bytes(b"")
    got = resolve_image_path(str(tmp_path), "images", "test.jpg")
    assert got == str(img_file)


def test_resolve_image_path_with_rig_subdir(tmp_path):
    """Test resolve_image_path preserving rig subdirectory."""
    images_dir = tmp_path / "images" / "rig-x"
    images_dir.mkdir(parents=True)
    img_file = images_dir / "test.jpg"
    img_file.write_bytes(b"")
    got = resolve_image_path(str(tmp_path), "images", "rig-x/test.jpg")
    assert got == str(img_file)


def test_resolve_image_path_rig_at_root(tmp_path):
    """Test resolve_image_path with rig at root."""
    rig_dir = tmp_path / "rig-x"
    rig_dir.mkdir()
    img_file = rig_dir / "test.jpg"
    img_file.write_bytes(b"")
    got = resolve_image_path(str(tmp_path), "images", "rig-x/test.jpg")
    assert got == str(img_file)


def test_resolve_depth_path_sibling_suffix_form(tmp_path):
    """Test resolve_depth_path finding sibling _depth.png form."""
    rig_dir = tmp_path / "rig-x"
    rig_dir.mkdir()
    img_file = rig_dir / "test.jpg"
    depth_file = rig_dir / "test_depth.png"
    img_file.write_bytes(b"")
    depth_file.write_bytes(b"")
    got = resolve_depth_path(str(img_file))
    assert got == str(depth_file)


def test_resolve_depth_path_standard_layout_flat(tmp_path):
    """Test resolve_depth_path finding depths/uuid.png (flat images)."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    depths_dir = tmp_path / "depths"
    depths_dir.mkdir()
    img_file = images_dir / "test.jpg"
    depth_file = depths_dir / "test.png"
    img_file.write_bytes(b"")
    depth_file.write_bytes(b"")
    got = resolve_depth_path(str(img_file))
    assert got == str(depth_file)


def test_resolve_depth_path_standard_layout_with_rig(tmp_path):
    """Test resolve_depth_path finding depths/rig-x/uuid.png (with rig preserved)."""
    images_rig_dir = tmp_path / "images" / "rig-x"
    images_rig_dir.mkdir(parents=True)
    depths_rig_dir = tmp_path / "depths" / "rig-x"
    depths_rig_dir.mkdir(parents=True)
    img_file = images_rig_dir / "test.jpg"
    depth_file = depths_rig_dir / "test.png"
    img_file.write_bytes(b"")
    depth_file.write_bytes(b"")
    got = resolve_depth_path(str(img_file))
    assert got == str(depth_file)


def test_resolve_depth_path_standard_layout_rig_fallback_flat(tmp_path):
    """Test resolve_depth_path falling back to depths/uuid.png when rig-specific not present."""
    images_rig_dir = tmp_path / "images" / "rig-x"
    images_rig_dir.mkdir(parents=True)
    depths_dir = tmp_path / "depths"
    depths_dir.mkdir()
    img_file = images_rig_dir / "test.jpg"
    depth_file = depths_dir / "test.png"
    img_file.write_bytes(b"")
    depth_file.write_bytes(b"")
    got = resolve_depth_path(str(img_file))
    assert got == str(depth_file)


def test_resolve_depth_path_returns_none_when_absent(tmp_path):
    """Test resolve_depth_path returns None when depth file does not exist."""
    p = tmp_path / "x.jpg"
    p.write_bytes(b"")
    assert resolve_depth_path(str(p)) is None


# detect_scene_type: the Scene.__init__ dispatch decision, factored out so it
# is testable without importing `scene` (which chains to the CUDA-only
# simple_knn._C extension via scene/gaussian_model.py).

def test_detect_scene_type_sparse_zero(tmp_path):
    """sparse/0 layout -> colmap."""
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.bin").write_bytes(b"")
    assert detect_scene_type(str(tmp_path)) == "colmap"


def test_detect_scene_type_sparse_only(tmp_path):
    """sparse/ (no /0) layout -> colmap."""
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    (sparse_dir / "cameras.bin").write_bytes(b"")
    assert detect_scene_type(str(tmp_path)) == "colmap"


def test_detect_scene_type_bare_zero(tmp_path):
    """0/ at the source root, no sparse/ wrapper -> colmap.

    This is the layout of the primary target dataset (input3/0/ holding the
    COLMAP model alongside sibling rig-*/ image+depth directories) and is
    exactly the case the old `os.path.exists(.../"sparse")` check missed.
    """
    zero_dir = tmp_path / "0"
    zero_dir.mkdir()
    (zero_dir / "cameras.bin").write_bytes(b"")
    assert detect_scene_type(str(tmp_path)) == "colmap"


def test_detect_scene_type_bare_root(tmp_path):
    """COLMAP model files directly at the source root -> colmap."""
    (tmp_path / "cameras.bin").write_bytes(b"")
    assert detect_scene_type(str(tmp_path)) == "colmap"


def test_detect_scene_type_blender(tmp_path):
    """No COLMAP model anywhere, but transforms_train.json present -> blender."""
    (tmp_path / "transforms_train.json").write_text("{}")
    assert detect_scene_type(str(tmp_path)) == "blender"


def test_detect_scene_type_neither(tmp_path):
    """Neither a COLMAP model nor transforms_train.json -> None."""
    assert detect_scene_type(str(tmp_path)) is None


def test_detect_scene_type_prefers_colmap_when_both_present(tmp_path):
    """A directory with both a COLMAP model and transforms_train.json is
    still colmap -- matches Scene.__init__'s original if/elif priority."""
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.bin").write_bytes(b"")
    (tmp_path / "transforms_train.json").write_text("{}")
    assert detect_scene_type(str(tmp_path)) == "colmap"
