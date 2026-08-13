"""Dense metric point-cloud initialisation from backprojected sensor depth.

Replaces COLMAP's sparse (~10k point) reconstruction with a dense
backprojection of every camera's depth map into a single world point cloud.
No CUDA dependency: the numeric work here is pure numpy.

`BasicPointCloud` (from `scene.gaussian_model`) is imported lazily inside
`backproject_cameras`, not at module scope, because `scene.gaussian_model`
chains to the `simple_knn._C` CUDA extension. That keeps this module -- and
`backproject_points`/`voxel_downsample` in particular -- importable and
testable on machines without a CUDA build.
"""

import numpy as np
from PIL import Image
from utils.graphics_utils import fov2focal


def voxel_downsample(points, colors, voxel_size):
    """Keep one point per occupied voxel."""
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return points[idx], colors[idx]


def backproject_points(cam_infos, scale_mm_per_unit, depth_max_m=30.0, stride=4):
    """Backproject every camera's depth map into world-space points and colors.

    Depth is planar Z, so x = (u - cx) * z / fx with no radial correction.
    Pure numpy, no CUDA-dependent imports -- this is the part that is
    directly unit-testable.

    Cameras whose `depth_path` is None are skipped. Invalid depth pixels
    (zero, or beyond `depth_max_m`) contribute no points.

    Returns:
        (points, colors) as float32 arrays of shape [N, 3]. Both are empty
        (shape [0, 3]) if no camera contributed any valid depth sample.
    """
    all_pts, all_cols = [], []
    for cam in cam_infos:
        if getattr(cam, "depth_path", None) is None:
            continue
        mm = np.asarray(Image.open(cam.depth_path)).astype(np.float32)
        H, W = mm.shape
        fx = fov2focal(cam.FovX, W)
        fy = fov2focal(cam.FovY, H)
        v, u = np.mgrid[0:H:stride, 0:W:stride]
        mm_s = mm[::stride, ::stride]
        z = mm_s / float(scale_mm_per_unit)
        keep = (mm_s > 0) & (mm_s < depth_max_m * 1000.0)
        u, v, z = u[keep], v[keep], z[keep]
        if len(z) == 0:
            continue
        x = (u - W * 0.5) * z / fx
        y = (v - H * 0.5) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)
        # cam.R is stored transposed (glm convention); world = pts_cam @ R.T + C
        C = -cam.R @ cam.T
        all_pts.append((pts_cam @ cam.R.T + C).astype(np.float32))

        rgb = np.asarray(Image.open(cam.image_path).convert("RGB"), dtype=np.float32) / 255.0
        all_cols.append(rgb[::stride, ::stride][keep])

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return np.concatenate(all_pts, 0), np.concatenate(all_cols, 0)


def backproject_cameras(cam_infos, scale_mm_per_unit, voxel_size,
                        depth_max_m=30.0, stride=4):
    """Backproject every camera's depth map into a single world BasicPointCloud."""
    from scene.gaussian_model import BasicPointCloud

    pts, cols = backproject_points(cam_infos, scale_mm_per_unit, depth_max_m, stride)
    pts, cols = voxel_downsample(pts, cols, voxel_size)
    return BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts))
