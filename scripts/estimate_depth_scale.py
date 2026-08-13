import json
import os
import struct
import sys
import numpy as np
from PIL import Image

# Allow running as `python3 scripts/estimate_depth_scale.py` from the repo
# root: Python only puts the script's own directory (scripts/) on sys.path,
# not the repo root, so `utils` would not otherwise be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.input_layout import find_model_dir, resolve_image_path, resolve_depth_path

MIN_POINTS_PER_IMAGE = 20
MIN_CONVENTION_MARGIN = 1.2

# Standalone COLMAP camera model table (id -> number of intrinsic params).
# Duplicated from scene/colmap_loader.py rather than imported: importing
# anything under `scene` executes scene/__init__.py, which chains to
# gaussian_model.py and imports the compiled simple_knn._C CUDA extension --
# unavailable on machines without a CUDA build (e.g. this one, macOS/no-torch).
# This script is stage-1 CPU pre-processing and must run without torch/CUDA.
_CAMERA_MODEL_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}


def qvec2rotmat(qvec):
    """Copied from scene/colmap_loader.py (see module docstring for why this
    file does not import from `scene`)."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_intrinsics_binary(path):
    """id -> camera dict with fx, fy, cx, cy. Standalone cameras.bin reader
    (see module docstring). Each record is (camera_id int32, model_id int32,
    width uint64, height uint64, params float64[num_params]).

    For 4-param models (e.g. PINHOLE) fx=params[0], fy=params[1]; for <=3
    params (e.g. SIMPLE_PINHOLE) fx=fy=params[0]. cx, cy are always the last
    two params.
    """
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            num_params = _CAMERA_MODEL_NUM_PARAMS[model_id]
            params = np.array(struct.unpack("<{}d".format(num_params), f.read(8 * num_params)))
            if len(params) > 3:
                fx, fy = params[0], params[1]
            else:
                fx = fy = params[0]
            cx, cy = params[-2], params[-1]
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }
    return cameras


def _read_points3D(path):
    """id -> xyz. colmap_loader drops the ids, and we need them."""
    pts = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid, x, y, z = struct.unpack("<Qddd", f.read(32))
            f.read(11)                                  # rgb + reprojection error
            ntrack = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * ntrack)
            pts[pid] = (x, y, z)
    return pts


def _read_images(path):
    """Full image records including keypoints and their point3D ids."""
    out = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            rec = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            dt = np.dtype([("x", "f8"), ("y", "f8"), ("id", "i8")])
            data = np.frombuffer(f.read(24 * npts), dtype=dt)
            out.append({
                "name": name.decode(),
                "q": rec[1:5],
                "t": rec[5:8],
                "camera_id": rec[8],
                "xy": np.stack([data["x"], data["y"]], 1),
                "ids": data["id"].copy(),
            })
    return out


def _mad(x):
    return float(np.median(np.abs(x - 1.0)))


def estimate_scale(source_path, images_arg="images", mad_mult=3.0):
    model_dir = find_model_dir(source_path)
    pts = _read_points3D(os.path.join(model_dir, "points3D.bin"))
    images = _read_images(os.path.join(model_dir, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(model_dir, "cameras.bin"))

    planar, radial, per_image = [], [], {}

    for im in images:
        try:
            img_path = resolve_image_path(source_path, images_arg, im["name"])
        except FileNotFoundError:
            continue
        dpath = resolve_depth_path(img_path)
        if dpath is None:
            continue

        cam = intr[im["camera_id"]]
        D = np.asarray(Image.open(dpath)).astype(np.float64)
        H, W = D.shape
        R, t = qvec2rotmat(np.array(im["q"])), np.array(im["t"])

        sel = im["ids"] > 0
        ids, xy = im["ids"][sel], im["xy"][sel]
        known = np.array([i in pts for i in ids], dtype=bool)
        if known.sum() < MIN_POINTS_PER_IMAGE:
            continue
        ids, xy = ids[known], xy[known]

        Pc = np.array([pts[i] for i in ids]) @ R.T + t
        z = Pc[:, 2]
        r = np.linalg.norm(Pc, axis=1)
        u = np.round(xy[:, 0]).astype(int)
        v = np.round(xy[:, 1]).astype(int)

        ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        z, r, u, v = z[ok], r[ok], u[ok], v[ok]
        if len(z) == 0:
            continue
        d = D[v, u]
        ok = d > 0
        z, r, d = z[ok], r[ok], d[ok]
        if len(z) < MIN_POINTS_PER_IMAGE:
            continue

        planar.append(d / z)
        radial.append(d / r)
        per_image[im["name"]] = float(np.median(d / z))

    if not planar:
        raise RuntimeError(
            "No usable (sparse point, depth pixel) samples under {}. "
            "Depth maps missing, or the model has too few points.".format(source_path))

    planar = np.concatenate(planar)
    radial = np.concatenate(radial)

    mad_planar = _mad(planar / np.median(planar))
    mad_radial = _mad(radial / np.median(radial))
    margin = mad_radial / max(mad_planar, 1e-9)

    if margin < MIN_CONVENTION_MARGIN:
        raise RuntimeError(
            "Depth convention is ambiguous or radial (planar MAD {:.4f}, radial MAD {:.4f}, "
            "margin {:.2f} < {:.2f}). This pipeline assumes planar Z; refusing to guess."
            .format(mad_planar, mad_radial, margin, MIN_CONVENTION_MARGIN))

    # Reject per-image outliers by MAD before taking the global scale. One image
    # in the reference capture reads 385 mm/unit against ~3940 -- a bad pose or
    # bad depth -- and must not drag the scale.
    names = list(per_image.keys())
    vals = np.array([per_image[n] for n in names])
    med = np.median(vals)
    spread = np.median(np.abs(vals - med)) or 1e-9
    outliers = [n for n, v in zip(names, vals) if abs(v - med) > mad_mult * spread]
    kept = np.array([v for n, v in zip(names, vals) if n not in outliers])

    return {
        "scale_mm_per_unit": float(np.median(kept)),
        "convention": "planar",
        "convention_margin": float(margin),
        "n_samples": int(planar.size),
        "per_image": per_image,
        "outliers": outliers,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("-i", "--images", default="images")
    ap.add_argument("--mad_mult", type=float, default=3.0)
    a = ap.parse_args()

    res = estimate_scale(a.source_path, a.images, a.mad_mult)
    out = os.path.join(find_model_dir(a.source_path), "depth_scale.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print("scale      : {:.1f} mm per COLMAP unit".format(res["scale_mm_per_unit"]))
    print("convention : {} (margin {:.2f}x)".format(res["convention"], res["convention_margin"]))
    print("samples    : {} over {} images".format(res["n_samples"], len(res["per_image"])))
    print("outliers   : {}".format(len(res["outliers"])))
    for n in res["outliers"]:
        print("   ! {}  ({:.0f} mm/unit)".format(n, res["per_image"][n]))
    print("written    : {}".format(out))
