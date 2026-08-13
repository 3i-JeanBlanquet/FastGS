import math
from collections import namedtuple

import numpy as np
from PIL import Image

from utils.depth_init import voxel_downsample, backproject_points


def test_voxel_downsample_collapses_duplicates():
    pts = np.array([[0.0, 0, 0], [0.001, 0, 0], [5.0, 0, 0]], dtype=np.float32)
    cols = np.ones((3, 3), dtype=np.float32)
    p, c = voxel_downsample(pts, cols, voxel_size=0.05)
    assert len(p) == 2, "two near-coincident points must merge, the far one must not"
    assert len(c) == 2


def test_voxel_downsample_preserves_all_when_spread_out():
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], dtype=np.float32)
    cols = np.ones((3, 3), dtype=np.float32)
    p, _ = voxel_downsample(pts, cols, voxel_size=0.05)
    assert len(p) == 3


# --- backproject_points: synthetic camera/depth tests -----------------------

StubCam = namedtuple("StubCam", ["depth_path", "image_path", "R", "T", "FovX", "FovY"])

# 8x8 image, fx = fy = 4 so that FovX = FovY = pi/2:
#   fov2focal(fov, pixels) = pixels / (2 * tan(fov / 2))
#   tan(fov/2) = pixels / (2 * fx) = 8 / (2 * 4) = 1  =>  fov = pi/2
_W = _H = 8
_FOV = math.pi / 2.0
_SCALE_MM_PER_UNIT = 1000.0  # so COLMAP units == metres
_DEPTH_MM = 2000  # 2.0 m


def _write_depth_png(path, arr):
    """Write a uint16 millimetre depth PNG with known values (matches the
    convention established in tests/test_depth_utils.py)."""
    Image.fromarray(arr.astype(np.uint16), mode="I;16").save(path)


def _write_rgb_png(path, shape=(_H, _W)):
    arr = np.zeros(shape + (3,), dtype=np.uint8)
    arr[..., 0] = 128  # arbitrary constant color, only used to check shape/count
    Image.fromarray(arr, mode="RGB").save(path)


def _make_cam(tmp_path, depth_arr, name="cam0", R=None, T=None):
    depth_path = str(tmp_path / (name + "_depth.png"))
    image_path = str(tmp_path / (name + ".png"))
    _write_depth_png(depth_path, depth_arr)
    _write_rgb_png(image_path)
    if R is None:
        R = np.eye(3, dtype=np.float64)
    if T is None:
        T = np.zeros(3, dtype=np.float64)
    return StubCam(depth_path=depth_path, image_path=image_path, R=R, T=T,
                   FovX=_FOV, FovY=_FOV)


def test_backproject_identity_camera_matches_hand_computed_points(tmp_path):
    # Identity rotation, zero translation => camera centre C = -R @ T = 0,
    # so world points equal camera-space points exactly.
    arr = np.full((_H, _W), _DEPTH_MM, dtype=np.uint16)
    cam = _make_cam(tmp_path, arr)

    pts, cols = backproject_points([cam], _SCALE_MM_PER_UNIT, depth_max_m=10.0, stride=4)

    # stride=4 over an 8x8 image samples u, v in {0, 4}.
    # z = 2000mm / 1000 (mm/unit) = 2.0 units for every sampled pixel.
    # x = (u - W/2) * z / fx = (u - 4) * 2.0 / 4 = (u - 4) * 0.5
    # y = (v - H/2) * z / fy = (v - 4) * 0.5  (fx == fy == 4, W == H == 8)
    expected = set()
    for u in (0, 4):
        for v in (0, 4):
            x = (u - 4) * 0.5
            y = (v - 4) * 0.5
            expected.add((x, y, 2.0))

    assert len(pts) == 4
    got = set((round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))
              for p in pts)
    assert got == expected
    assert len(cols) == 4


def test_backproject_zero_depth_pixels_produce_no_points(tmp_path):
    arr = np.full((_H, _W), _DEPTH_MM, dtype=np.uint16)
    # Zero out one of the 4 sampled grid points (u=4, v=4) at stride=4.
    arr[4, 4] = 0
    cam = _make_cam(tmp_path, arr)

    pts, cols = backproject_points([cam], _SCALE_MM_PER_UNIT, depth_max_m=10.0, stride=4)

    assert len(pts) == 3, "the zeroed sample must be dropped, the other 3 kept"
    assert len(cols) == 3
    # None of the surviving points should be the one that would have come
    # from the zeroed (u=4, v=4) pixel: x = (4-4)*0.5 = 0, y = (4-4)*0.5 = 0.
    assert not any(
        abs(float(p[0])) < 1e-9 and abs(float(p[1])) < 1e-9 for p in pts
    )


def test_backproject_all_invalid_depth_produces_empty_arrays(tmp_path):
    arr = np.zeros((_H, _W), dtype=np.uint16)
    cam = _make_cam(tmp_path, arr)

    pts, cols = backproject_points([cam], _SCALE_MM_PER_UNIT, depth_max_m=10.0, stride=4)

    assert pts.shape == (0, 3)
    assert cols.shape == (0, 3)


def test_backproject_skips_camera_with_no_depth_path(tmp_path):
    arr = np.full((_H, _W), _DEPTH_MM, dtype=np.uint16)
    valid_cam = _make_cam(tmp_path, arr, name="valid")
    skipped_cam = StubCam(depth_path=None, image_path=valid_cam.image_path,
                          R=np.eye(3), T=np.zeros(3), FovX=_FOV, FovY=_FOV)

    pts_both, _ = backproject_points([skipped_cam, valid_cam], _SCALE_MM_PER_UNIT,
                                     depth_max_m=10.0, stride=4)
    pts_valid_only, _ = backproject_points([valid_cam], _SCALE_MM_PER_UNIT,
                                           depth_max_m=10.0, stride=4)

    assert len(pts_both) == len(pts_valid_only) == 4, \
        "camera with depth_path=None must contribute zero points"


def test_backproject_no_cameras_returns_empty_arrays():
    pts, cols = backproject_points([], _SCALE_MM_PER_UNIT, depth_max_m=10.0, stride=4)
    assert pts.shape == (0, 3)
    assert cols.shape == (0, 3)


def test_backproject_translated_camera_offsets_world_points(tmp_path):
    # R = identity, T = (1, 2, 3) => world = -T (camera centre) + cam-space point
    # (since R.T == R == I): world = pts_cam - T.
    arr = np.full((_H, _W), _DEPTH_MM, dtype=np.uint16)
    T = np.array([1.0, 2.0, 3.0])
    cam = _make_cam(tmp_path, arr, R=np.eye(3), T=T)

    pts, _ = backproject_points([cam], _SCALE_MM_PER_UNIT, depth_max_m=10.0, stride=4)

    # Pick the (u=0, v=0) sample: cam-space point is (-2.0, -2.0, 2.0).
    # world = pts_cam @ R.T + C, C = -R @ T = -T = (-1, -2, -3)
    # world = (-2.0, -2.0, 2.0) + (-1, -2, -3) = (-3.0, -4.0, -1.0)
    expected = (-3.0, -4.0, -1.0)
    got = set((round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))
              for p in pts)
    assert expected in got
