# Depth Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Supervise FastGS with metric sensor depth so the exported PLY has correct geometry instead of floaters and ray-smeared splats.

**Architecture:** Three stages. A CPU calibration pass measures the per-scene COLMAP-to-millimetre scale and asserts the depth convention. The CUDA rasterizer gains one accumulated channel (expected depth) plus an alpha output, with gradients routed to both Gaussian view-space z and alpha. Python does the rest: loading, masking, and dn-splatter's `EdgeAwareLogL1` loss.

**Tech Stack:** Python 3.7.13, PyTorch 1.12.1+cu116, CUDA 11.6, numpy, Pillow, pytest. CUDA extension built with `sm_86 + PTX`.

**Spec:** `docs/superpowers/specs/2026-08-13-depth-supervision-design.md`

## Global Constraints

- **Build env:** all torch/CUDA code must compile under **torch 1.12.1 / CUDA 11.6**, inside the `fastgs` conda env in the `rnd-3dgs` container. Do NOT use torch 2.x-only APIs.
- **GPU arch:** build `TORCH_CUDA_ARCH_LIST="8.6+PTX"`. CUDA 11.6 cannot target `sm_89`; the RTX 6000 Ada runs the PTX JIT path.
- **Depth convention:** planar Z, along the optical axis. Measured (scene 1 MAD 2.80 % vs radial 10.07 %; scene 2 residual correlation +0.025 vs radial −0.464). Never apply a radial correction.
- **Depth units:** uint16 PNG in **millimetres**, `0 = invalid`. Confirmed by `dn_splatter` `depth_unit_scale_factor = 0.001`.
- **Scale is per-scene and must be measured.** Scene 1 = 3939.7 mm/unit, scene 2 = 4135.3 mm/unit — 5.0 % apart. Never hardcode; never default to 4.0.
- **Loss parity:** match dn-splatter's tuned preset — `EdgeAwareLogL1`, `depth_lambda = 0.2`.
- **Upstream style:** FastGS is vanilla INRIA-derived code with no test suite and no type annotations. Match the surrounding style; do not reformat existing files.
- **Never modify** anything under `submodules/*/third_party/`.

## Reference data (on the dev Mac)

| name | COLMAP model | images root | images | expected scale |
|---|---|---|---|---|
| `SCENE_MESH` | `.../rnd.project.mesh/test/input3/0` | `.../input3` | 112 | ≈4135 mm/unit |
| `SCENE_RECON` | `.../rnd.project.reconstruction/test/dest/0` | `.../test/source/images` | 176 | ≈3940 mm/unit |

Full prefix for both: `/Users/3i-a1-2025-003/Documents/repositories/rnd.toolkit/pipeline/`

Tasks 1–2 run on the Mac (numpy + Pillow only). Tasks 3+ need torch, so they run in the container:
`ssh ubuntu@172.17.0.3` then `docker exec rnd-3dgs conda run -n fastgs ...`

## File Structure

| file | responsibility |
|---|---|
| `utils/input_layout.py` (new) | Resolve model dir, image paths, depth paths across both on-disk layouts |
| `utils/depth_utils.py` (new) | Load/mask/scale a depth PNG into a tensor |
| `scripts/estimate_depth_scale.py` (new) | Stage 1 calibration → `depth_scale.json` |
| `scripts/eval_depth.py` (new) | Depth error metrics on held-out views |
| `scene/dataset_readers.py` | Carry depth paths on `CameraInfo`; fix rig-subdir path bug |
| `utils/camera_utils.py` | Pass depth through to `Camera` |
| `scene/cameras.py` | Store `sensor_depth` + `depth_mask` |
| `utils/loss_utils.py` | `edge_aware_logl1_loss` |
| `gaussian_renderer/__init__.py` | Return `depth` and `alpha` |
| `submodules/.../cuda_rasterizer/{forward,backward}.{cu,h}` | Render + backprop depth |
| `submodules/.../{rasterizer.h,rasterizer_impl.cu,rasterize_points.cu,ext.cpp}` | Thread buffers through |
| `submodules/.../diff_gaussian_rasterization_fastgs/__init__.py` | Expose depth in the autograd Function |
| `train.py` | Add depth term to the loss |
| `scene/gaussian_model.py`, `scene/__init__.py` | Optional metric PLY export |
| `arguments/__init__.py` | New CLI flags |
| `tests/` (new) | pytest suite |

---

### Task 1: Input layout resolver

Two on-disk layouts exist. The reconstruction output puts rig folders at the root with depth as a sibling (`rig-1,0,0,0/<uuid>_depth.png`) and the COLMAP model in `0/`. The standard 3DGS layout uses `images/`, `depths/`, `sparse/0/`. Both must work.

This also fixes a real bug: `dataset_readers.py:97` calls `os.path.basename(extr.name)`, which strips the rig folder — so all four faces of a station collapse onto one path and the scene silently loads wrong images.

**Files:**
- Create: `utils/input_layout.py`
- Create: `tests/test_input_layout.py`

**Interfaces:**
- Produces: `find_model_dir(source_path) -> str`, `resolve_image_path(source_path, images_arg, colmap_name) -> str`, `resolve_depth_path(image_path) -> str or None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_input_layout.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/3i-a1-2025-003/Documents/repositories/fastgs/FastGS && python3 -m pytest tests/test_input_layout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.input_layout'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/input_layout.py
import os

# COLMAP model may sit in the standard 3DGS location or bare at the input root.
_MODEL_CANDIDATES = [os.path.join("sparse", "0"), "sparse", "0", ""]


def find_model_dir(source_path):
    """Return the directory holding cameras/images/points3D (.bin or .txt)."""
    for rel in _MODEL_CANDIDATES:
        d = os.path.join(source_path, rel) if rel else source_path
        if os.path.exists(os.path.join(d, "cameras.bin")) or \
           os.path.exists(os.path.join(d, "cameras.txt")):
            return d
    raise FileNotFoundError(
        "No COLMAP model under {}; tried {}".format(source_path, _MODEL_CANDIDATES))


def resolve_image_path(source_path, images_arg, colmap_name):
    """Resolve a COLMAP image name to a path on disk.

    colmap_name may carry a rig subdirectory ("rig-1,0,0,0/<uuid>.jpg"), which
    must be preserved -- the four cube faces of a station share a basename and
    would otherwise collapse onto one file.
    """
    reading_dir = images_arg if images_arg else "images"
    candidates = [
        os.path.join(source_path, colmap_name),                  # rig dirs at root
        os.path.join(source_path, reading_dir, colmap_name),     # rig dirs under images/
        os.path.join(source_path, reading_dir, os.path.basename(colmap_name)),  # flat
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Image {} not found; tried:\n  {}".format(colmap_name, "\n  ".join(candidates)))


def resolve_depth_path(image_path):
    """Return the depth PNG for an image, or None if there is not one."""
    stem, _ = os.path.splitext(image_path)
    d = os.path.dirname(image_path)
    base = os.path.basename(stem)
    candidates = [
        stem + "_depth.png",                                   # sibling suffix form
        os.path.join(os.path.dirname(d), "depths", base + ".png"),  # images/ + depths/
        os.path.join(d, "..", "depths", base + ".png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.normpath(c)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_input_layout.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add utils/input_layout.py tests/test_input_layout.py
git commit -m "feat: resolve input layouts and preserve rig subdirectories

os.path.basename in readColmapCameras collapsed all four cube faces of a
station onto one path. Resolve against the full COLMAP name instead, and
accept both the rig-at-root and images/+depths/ layouts."
```

---

### Task 2: Calibration script

Measures the per-scene scale and asserts the convention. This is the stage that prevents a ~4000x gradient mismatch, so its assertions are hard failures, not warnings.

**Files:**
- Create: `scripts/estimate_depth_scale.py`
- Create: `tests/test_estimate_depth_scale.py`

**Interfaces:**
- Consumes: `find_model_dir`, `resolve_image_path`, `resolve_depth_path` (Task 1)
- Produces: `estimate_scale(source_path, images_arg="images", mad_mult=3.0) -> dict` with keys `scale_mm_per_unit`, `convention`, `convention_margin`, `n_samples`, `per_image`, `outliers`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_estimate_depth_scale.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_estimate_depth_scale.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.estimate_depth_scale'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/__init__.py` (empty) and `scripts/estimate_depth_scale.py`:

```python
# scripts/estimate_depth_scale.py
import json
import os
import struct
import numpy as np
from PIL import Image

from scene.colmap_loader import read_intrinsics_binary, qvec2rotmat
from utils.input_layout import find_model_dir, resolve_image_path, resolve_depth_path

MIN_POINTS_PER_IMAGE = 20
MIN_CONVENTION_MARGIN = 1.2


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_estimate_depth_scale.py -v`
Expected: 3 passed

Then confirm the CLI on real data:
Run: `python3 scripts/estimate_depth_scale.py -s "$PREFIX/rnd.project.mesh/test/input3"`
Expected: scale ≈4135, convention planar, `depth_scale.json` written into `input3/0/`

- [ ] **Step 5: Commit**

```bash
git add scripts/ tests/test_estimate_depth_scale.py
git commit -m "feat: measure per-scene depth scale and assert planar convention

COLMAP poses are not metric (3939.7 vs 4135.3 mm/unit across two
captures). Supervising millimetre depth without this scale puts a ~4000x
mismatch into the gradient. Fails loudly rather than guessing."
```

---

### Task 3: Load depth onto cameras

**Files:**
- Create: `utils/depth_utils.py`
- Create: `tests/test_depth_utils.py`
- Modify: `scene/dataset_readers.py` (`CameraInfo` at 26-36; `readColmapCameras` at 68-105; `readColmapSceneInfo` at 132-177)
- Modify: `utils/camera_utils.py` (`loadCam` at 19-52)
- Modify: `scene/cameras.py` (`Camera.__init__` at 18-57)

**Interfaces:**
- Consumes: `resolve_depth_path` (Task 1), `depth_scale.json` (Task 2)
- Produces: `load_depth(path, scale_mm_per_unit, resolution, depth_max_m) -> (torch.FloatTensor [1,H,W], torch.BoolTensor [1,H,W])`; `Camera.sensor_depth`, `Camera.depth_mask` (both `None` when depth is absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depth_utils.py
import os
import torch
import pytest
from utils.depth_utils import load_depth

PREFIX = "/Users/3i-a1-2025-003/Documents/repositories/rnd.toolkit/pipeline"
DEPTH = os.path.join(PREFIX, "rnd.project.mesh/test/input3",
                     "rig-1,0,0,0/0179c0d4-f4d6-4003-b156-dbd7b2d3e8a9_depth.png")

pytestmark = pytest.mark.skipif(not os.path.exists(DEPTH), reason="reference data absent")
SCALE = 4135.3


def test_shape_and_dtype():
    d, m = load_depth(DEPTH, SCALE, (1024, 1024), 30.0)
    assert d.shape == (1, 1024, 1024) and d.dtype == torch.float32
    assert m.shape == (1, 1024, 1024) and m.dtype == torch.bool


def test_zero_pixels_are_masked_out():
    d, m = load_depth(DEPTH, SCALE, (1024, 1024), 30.0)
    assert m.sum() > 0
    assert not torch.isnan(d).any()


def test_values_land_in_colmap_units_not_millimetres():
    # A ~7 m median at 4135 mm/unit is ~1.7 COLMAP units.
    d, m = load_depth(DEPTH, SCALE, (1024, 1024), 30.0)
    med = d[m].median().item()
    assert 0.2 < med < 8.0, "expected COLMAP units, got {}".format(med)


def test_depth_max_rejects_far_pixels():
    _, m_far = load_depth(DEPTH, SCALE, (1024, 1024), 30.0)
    _, m_near = load_depth(DEPTH, SCALE, (1024, 1024), 3.0)
    assert m_near.sum() < m_far.sum()


def test_resize_matches_requested_resolution():
    d, m = load_depth(DEPTH, SCALE, (512, 512), 30.0)
    assert d.shape == (1, 512, 512) and m.shape == (1, 512, 512)
```

- [ ] **Step 2: Run test to verify it fails**

Run (in the container's fastgs env, since torch is required):
`docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/test_depth_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.depth_utils'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/depth_utils.py
import numpy as np
import torch
from PIL import Image


def load_depth(path, scale_mm_per_unit, resolution, depth_max_m):
    """Load a uint16 millimetre depth PNG as (depth_in_colmap_units, valid_mask).

    Masking happens in millimetres so the thresholds stay physical; conversion
    to COLMAP units comes after. Depth is planar Z -- no radial correction.
    """
    img = Image.open(path)
    if img.size != tuple(resolution):
        # NEAREST: interpolating across a depth discontinuity invents surfaces
        # that exist at neither depth.
        img = img.resize(resolution, Image.NEAREST)
    mm = np.asarray(img).astype(np.float32)

    valid = (mm > 0) & (mm < depth_max_m * 1000.0)
    units = mm / float(scale_mm_per_unit)
    units[~valid] = 0.0

    depth = torch.from_numpy(units).unsqueeze(0).contiguous()
    mask = torch.from_numpy(valid).unsqueeze(0).contiguous()
    return depth, mask
```

Then thread it through:

- `scene/dataset_readers.py`: add `depth_path: str` to `CameraInfo` (default `None`). In `readColmapCameras`, replace the `os.path.basename(extr.name)` join at line 97 with `resolve_image_path(...)`, and set `depth_path=resolve_depth_path(image_path)`. Pass `source_path` into `readColmapCameras`. In `readColmapSceneInfo`, replace the hardcoded `sparse/0` paths with `find_model_dir(path)`, and load `depth_scale.json` from the model dir if present, storing it on `SceneInfo` as a new `depth_scale` field (`None` when absent).
- `utils/camera_utils.py`: in `loadCam`, when `args.depths` is set and `cam_info.depth_path` is not `None`, call `load_depth(...)` with the already-computed `resolution` and pass the result into `Camera`.
- `scene/cameras.py`: accept `sensor_depth=None, depth_mask=None` in `__init__` and store them:

```python
self.sensor_depth = None
self.depth_mask = None
if sensor_depth is not None:
    # fp16 depth + bool mask is ~550 MB across 176 cameras at 1024^2, on top of
    # the ~2.2 GB the RGB images already hold. --depth_on_cpu trades latency for
    # VRAM on smaller cards.
    dev = torch.device("cpu") if depth_on_cpu else self.data_device
    self.sensor_depth = sensor_depth.to(dev).half()
    self.depth_mask = depth_mask.to(dev)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/ -v`
Expected: all pass, including Tasks 1–2 still green

- [ ] **Step 5: Commit**

```bash
git add utils/depth_utils.py tests/test_depth_utils.py scene/ utils/camera_utils.py
git commit -m "feat: load sensor depth onto cameras

Masks in millimetres before converting to COLMAP units so thresholds stay
physical. Nearest-neighbour resize only -- interpolating across a depth
discontinuity invents surfaces at neither depth."
```

---

### Task 4: Dense depth initialisation

No CUDA needed, and valuable on its own: replaces COLMAP's ~10k sparse points with a dense metric backprojection.

**Files:**
- Create: `utils/depth_init.py`
- Create: `tests/test_depth_init.py`
- Modify: `scene/dataset_readers.py` (`readColmapSceneInfo`)
- Modify: `arguments/__init__.py` (`ModelParams`)

**Interfaces:**
- Consumes: `CameraInfo.depth_path`, `SceneInfo.depth_scale` (Task 3)
- Produces: `backproject_cameras(cam_infos, scale_mm_per_unit, voxel_size, depth_max_m, stride) -> BasicPointCloud`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depth_init.py
import numpy as np
from utils.depth_init import voxel_downsample


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_depth_init.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.depth_init'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/depth_init.py
import numpy as np
from PIL import Image
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import fov2focal


def voxel_downsample(points, colors, voxel_size):
    """Keep one point per occupied voxel."""
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return points[idx], colors[idx]


def backproject_cameras(cam_infos, scale_mm_per_unit, voxel_size,
                        depth_max_m=30.0, stride=4):
    """Backproject every camera's depth map into a single world point cloud.

    Depth is planar Z, so x = (u - cx) * z / fx with no radial correction.
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
        z = mm[::stride, ::stride] / float(scale_mm_per_unit)
        keep = (mm[::stride, ::stride] > 0) & (mm[::stride, ::stride] < depth_max_m * 1000.0)
        u, v, z = u[keep], v[keep], z[keep]
        if len(z) == 0:
            continue
        x = (u - W * 0.5) * z / fx
        y = (v - H * 0.5) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)
        # cam.R is stored transposed (glm convention); world = R @ p_cam + C
        C = -cam.R @ cam.T
        all_pts.append(pts_cam @ cam.R.T + C)

        rgb = np.asarray(Image.open(cam.image_path).convert("RGB"), dtype=np.float32) / 255.0
        all_cols.append(rgb[::stride, ::stride][keep])

    pts = np.concatenate(all_pts, 0)
    cols = np.concatenate(all_cols, 0)
    pts, cols = voxel_downsample(pts, cols, voxel_size)
    return BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts))
```

In `readColmapSceneInfo`, when `args.init_from_depth` is set and a scale is available, use this instead of `fetchPly`. Add `self.init_from_depth = False` and `self.depth_init_voxel = 0.02` to `ModelParams`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_depth_init.py -v`
Expected: 2 passed

Sanity-check on real data — point count should land in the low millions before downsampling and drop substantially after:
`python3 -c "from utils.depth_init import *; ..."`

- [ ] **Step 5: Commit**

```bash
git add utils/depth_init.py tests/test_depth_init.py scene/dataset_readers.py arguments/__init__.py
git commit -m "feat: initialise from dense backprojected depth

Replaces ~10k COLMAP sparse points with a dense metric backprojection.
Independent of the CUDA work, so it lands even if the kernels slip."
```

---

### Task 5: CUDA forward — render depth and alpha

`GeometryState.depths` (`rasterizer_impl.h:32`) already holds per-Gaussian view-space z for sorting, so the forward pass only needs to accumulate it.

**Files:**
- Modify: `cuda_rasterizer/forward.cu` (`renderCUDA` at 275-438, `FORWARD::render` at 441-484)
- Modify: `cuda_rasterizer/forward.h`, `cuda_rasterizer/rasterizer.h` (`forward` at 31-59)
- Modify: `cuda_rasterizer/rasterizer_impl.cu` (pass `geomState.depths`; widen `sampled_ar` to `CHANNELS + 1`)
- Modify: `rasterize_points.cu`, `ext.cpp`
- Modify: `submodules/.../diff_gaussian_rasterization_fastgs/__init__.py`
- Create: `tests/test_render_depth.py`

**Interfaces:**
- Produces: `_RasterizeGaussians.forward` returns `(color, radii, accum_metric_counts, depth, alpha)`; `GaussianRasterizer.forward` returns the same 5-tuple

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render_depth.py
import math
import torch
import pytest
from diff_gaussian_rasterization_fastgs import (
    GaussianRasterizationSettings, GaussianRasterizer)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _settings(W=64, H=64):
    fov = math.pi / 2
    tan = math.tan(fov * 0.5)
    view = torch.eye(4, device="cuda")
    znear, zfar = 0.01, 100.0
    proj = torch.zeros(4, 4, device="cuda")
    proj[0, 0] = 1.0 / tan
    proj[1, 1] = 1.0 / tan
    proj[2, 2] = zfar / (zfar - znear)
    proj[3, 2] = -(zfar * znear) / (zfar - znear)
    proj[2, 3] = 1.0
    return GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tan, tanfovy=tan,
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=view, projmatrix=view @ proj, sh_degree=0,
        campos=torch.zeros(3, device="cuda"), mult=0.5, prefiltered=False,
        debug=False, get_flag=False,
        metric_map=torch.zeros(H * W, dtype=torch.int, device="cuda"))


def _one_gaussian(z, opacity=0.99, scale=0.05):
    return dict(
        means3D=torch.tensor([[0.0, 0.0, z]], device="cuda"),
        means2D=torch.zeros((1, 4), device="cuda"),
        opacities=torch.tensor([[opacity]], device="cuda"),
        scales=torch.full((1, 3), scale, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        colors_precomp=torch.ones((1, 3), device="cuda"))


def test_single_opaque_gaussian_renders_its_own_depth():
    r = GaussianRasterizer(_settings())
    out = r(**_one_gaussian(z=3.0))
    depth, alpha = out[3], out[4]
    covered = alpha > 0.9
    assert covered.sum() > 0, "the splat must cover some pixels"
    assert torch.allclose(depth[covered], torch.full_like(depth[covered], 3.0), atol=0.15)


def test_depth_tracks_distance():
    r = GaussianRasterizer(_settings())
    d_near = r(**_one_gaussian(z=2.0))[3]
    d_far = r(**_one_gaussian(z=6.0))[3]
    assert d_far.max() > d_near.max() + 1.0


def test_alpha_is_zero_where_nothing_is_drawn():
    r = GaussianRasterizer(_settings())
    alpha = r(**_one_gaussian(z=3.0))[4]
    assert alpha.min() < 0.01, "empty pixels must have zero coverage"
```

- [ ] **Step 2: Run test to verify it fails**

Build then run:
```bash
docker exec rnd-3dgs bash -lc 'cd /app/gaussian-splatting/FastGS/submodules/diff-gaussian-rasterization_fastgs && TORCH_CUDA_ARCH_LIST="8.6+PTX" conda run -n fastgs pip install -e . '
docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/test_render_depth.py -v
```
Expected: FAIL — the rasterizer returns a 3-tuple, so `out[3]` raises `IndexError`

- [ ] **Step 3: Write minimal implementation**

In `forward.cu::renderCUDA`, add three parameters and one accumulator:

```cuda
// signature (after `const float* __restrict__ features`)
const float* __restrict__ depths,
// ... and alongside the existing outputs:
float* __restrict__ out_depth,
float* __restrict__ out_alpha,

// with the other accumulators (near `float C[CHANNELS] = { 0 };`)
float D = 0.0f;

// in the per-32-Gaussian bucket checkpoint (currently lines 363-369),
// store the depth accumulator in the extra slot so backward can replay it
if (j % 32 == 0) {
    sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;
    for (int ch = 0; ch < CHANNELS; ++ch) {
        sampled_ar[(bbm * BLOCK_SIZE * (CHANNELS + 1)) + ch * BLOCK_SIZE + block.thread_rank()] = C[ch];
    }
    sampled_ar[(bbm * BLOCK_SIZE * (CHANNELS + 1)) + CHANNELS * BLOCK_SIZE + block.thread_rank()] = D;
    ++bbm;
}

// in the accumulation block (currently lines 398-399)
for (int ch = 0; ch < CHANNELS; ch++)
    C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
D += depths[collected_id[j]] * alpha * T;

// in the writeout block (currently lines 420-429)
if (inside)
{
    final_T[pix_id] = T;
    n_contrib[pix_id] = last_contributor;
    out_depth[pix_id] = D;
    out_alpha[pix_id] = 1.0f - T;   // gradient-free; masking and diagnostics only
    for (int ch = 0; ch < CHANNELS; ch++) { /* unchanged */ }
}
```

Every other `sampled_ar` index in the codebase must move from a stride of
`CHANNELS` to `CHANNELS + 1` — that includes the allocation in
`rasterizer_impl.cu` and the replay in `backward.cu` (`Shared_sampled_ar`,
currently declared `[32 * C + 1]` at line 493, and the offset at line 494).
**A missed stride update is the most likely bug in this task** and shows up as
colour channels bleeding into each other rather than as a depth error.

Pass `geomState.depths` into `FORWARD::render`, and thread `out_depth` /
`out_alpha` through `FORWARD::render` (`forward.h`) and `Rasterizer::forward`
(`rasterizer.h:31-59`).

In `rasterize_points.cu`, allocate both outputs and return them:

```cpp
torch::Tensor out_depth = torch::full({H, W}, 0.0, float_opts);
torch::Tensor out_alpha = torch::full({H, W}, 0.0, float_opts);
// ... pass out_depth.contiguous().data_ptr<float>() etc. into the rasterizer
return std::make_tuple(rendered, num_buckets, out_color, radii, geomBuffer,
                       binningBuffer, imgBuffer, sampleBuffer,
                       accum_metric_counts, out_depth, out_alpha);
```

Update the Python binding in `diff_gaussian_rasterization_fastgs/__init__.py`:
unpack the two extra values in `forward`, return
`color, radii, accum_metric_counts, depth, alpha`, and widen `backward`'s
signature to accept the matching extra incoming gradients — ignore them for now
(Task 6 wires the depth one; alpha's stays permanently unused):

```python
@staticmethod
def backward(ctx, grad_out_color, _, g_metric, grad_out_depth, grad_out_alpha):
```

- [ ] **Step 4: Run test to verify it passes**

Run: rebuild, then `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/test_render_depth.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add submodules/diff-gaussian-rasterization_fastgs tests/test_render_depth.py
git commit -m "feat(cuda): render expected depth and alpha

Accumulates D = sum(d_i * alpha_i * T_i) alongside colour, and exposes the
already-computed final_T as alpha. Alpha is gradient-free -- it is used for
masking only."
```

---

### Task 6: CUDA backward — depth gradients

Both gradient paths matter and serve different purposes. The position path slides Gaussians along the ray onto the surface; **the alpha path is what removes floaters** — a splat in front of a wall makes rendered depth too small, and its opacity gets pushed down. Implementing only the position path leaves the headline problem unfixed.

**Files:**
- Modify: `cuda_rasterizer/backward.cu` (per-bucket kernel at 403-620; per-pixel `renderCUDA` at 623+; `preprocessCUDA` at 349)
- Modify: `cuda_rasterizer/backward.h`, `cuda_rasterizer/rasterizer.h` (`backward` at 61-93)
- Modify: `rasterize_points.cu`, Python binding
- Create: `tests/test_depth_backward.py`

**Interfaces:**
- Consumes: forward's `out_depth` (Task 5)
- Produces: `dL_dmeans3D` and `dL_dopacity` contributions from `dL_ddepth`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depth_backward.py
import torch
import pytest
from tests.test_render_depth import _settings, _one_gaussian
from diff_gaussian_rasterization_fastgs import GaussianRasterizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rendered_depth_sum(z_value, opacity=0.99):
    g = _one_gaussian(z=z_value, opacity=opacity)
    g["means3D"] = g["means3D"].clone().requires_grad_(True)
    out = GaussianRasterizer(_settings())(**g)
    return out[3].sum(), g["means3D"]


def test_depth_gradient_flows_to_position():
    total, means = _rendered_depth_sum(3.0)
    total.backward()
    assert means.grad is not None
    assert means.grad[0, 2].abs() > 1e-6, "dL/dz must be non-zero"


def test_position_gradient_matches_finite_difference():
    eps = 0.05
    total, means = _rendered_depth_sum(3.0)
    total.backward()
    analytic = means.grad[0, 2].item()
    with torch.no_grad():
        hi = _rendered_depth_sum(3.0 + eps)[0].item()
        lo = _rendered_depth_sum(3.0 - eps)[0].item()
    numeric = (hi - lo) / (2 * eps)
    assert abs(analytic - numeric) / max(abs(numeric), 1e-3) < 0.05


def test_depth_gradient_flows_to_opacity():
    g = _one_gaussian(z=3.0)
    g["opacities"] = g["opacities"].clone().requires_grad_(True)
    out = GaussianRasterizer(_settings())(**g)
    out[3].sum().backward()
    assert g["opacities"].grad is not None
    assert g["opacities"].grad.abs().sum() > 1e-6, \
        "the alpha path is what removes floaters -- it must carry gradient"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/test_depth_backward.py -v`
Expected: FAIL — depth output has no `grad_fn` wired, so `.backward()` raises or grads stay `None`

- [ ] **Step 3: Write minimal implementation**

Mirror the existing colour derivation with a depth channel and zero background. In the per-bucket kernel, replay `D` from the widened `sampled_ar` exactly as `ar[ch]` is replayed for colour (line 539), then:

```cuda
// depth channel: same structure as colour, background = 0
const float dL_dD = dL_ddepths_pix;               // per-pixel incoming grad
dL_dalpha += (d_i * T + one_minus_alpha_reci * ar_depth) * dL_dD;
atomicAdd(&dL_ddepths[gaussian_idx], alpha * T * dL_dD);
```

In `preprocessCUDA`, route the accumulated per-Gaussian depth gradient to position. Since `z_view = V[2]*x + V[6]*y + V[10]*z + V[14]`:

```cuda
dL_dmean3D[idx].x += dL_ddepth * viewmatrix[2];
dL_dmean3D[idx].y += dL_ddepth * viewmatrix[6];
dL_dmean3D[idx].z += dL_ddepth * viewmatrix[10];
```

Allocate a `dL_ddepths` buffer of `P` floats in `rasterize_points.cu`, thread `dL_dout_depth` through `Rasterizer::backward`, and pass `grad_depth` from the Python `backward` (it arrives as the 4th incoming gradient).

- [ ] **Step 4: Run test to verify it passes**

Run: rebuild, then `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/test_depth_backward.py tests/test_render_depth.py -v`
Expected: all pass. If the finite-difference test fails by a constant factor, the bucket replay of the depth accumulator is misindexed — check the `CHANNELS` offset in `sampled_ar`.

- [ ] **Step 5: Commit**

```bash
git add submodules/diff-gaussian-rasterization_fastgs tests/test_depth_backward.py
git commit -m "feat(cuda): backpropagate depth to position and alpha

Verified against finite differences. The alpha path is the one that
removes floaters: a splat in front of a wall renders depth too small and
gets its opacity pushed down."
```

---

### Task 7: Loss and training integration

**Files:**
- Modify: `utils/loss_utils.py`
- Modify: `gaussian_renderer/__init__.py` (return dict at 106-110)
- Modify: `train.py` (loss at 99-104)
- Modify: `arguments/__init__.py`
- Create: `tests/test_depth_loss.py`

**Interfaces:**
- Consumes: `render_fastgs(...)["depth"]`, `Camera.sensor_depth`, `Camera.depth_mask`
- Produces: `edge_aware_logl1_loss(pred, gt, rgb, mask) -> scalar tensor`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depth_loss.py
import torch
from utils.loss_utils import edge_aware_logl1_loss


def test_zero_when_prediction_is_exact():
    d = torch.rand(1, 16, 16) + 1.0
    rgb = torch.rand(3, 16, 16)
    m = torch.ones(1, 16, 16, dtype=torch.bool)
    assert edge_aware_logl1_loss(d, d, rgb, m).item() < 1e-6


def test_masked_pixels_are_ignored():
    gt = torch.ones(1, 8, 8)
    pred = gt.clone()
    pred[0, 0, 0] = 99.0
    rgb = torch.zeros(3, 8, 8)
    m = torch.ones(1, 8, 8, dtype=torch.bool)
    m[0, 0, 0] = False
    assert edge_aware_logl1_loss(pred, gt, rgb, m).item() < 1e-6


def test_edges_are_down_weighted():
    gt = torch.ones(1, 8, 16)
    pred = gt + 1.0
    m = torch.ones(1, 8, 16, dtype=torch.bool)
    flat = torch.zeros(3, 8, 16)
    edgy = torch.zeros(3, 8, 16)
    edgy[:, :, 8:] = 1.0        # a hard vertical edge
    assert edge_aware_logl1_loss(pred, gt, edgy, m) < edge_aware_logl1_loss(pred, gt, flat, m)


def test_log_compresses_large_errors():
    gt = torch.ones(1, 8, 8)
    rgb = torch.zeros(3, 8, 8)
    m = torch.ones(1, 8, 8, dtype=torch.bool)
    small = edge_aware_logl1_loss(gt + 1.0, gt, rgb, m).item()
    huge = edge_aware_logl1_loss(gt + 100.0, gt, rgb, m).item()
    assert huge < 20 * small, "log must keep outlier pixels from dominating"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/test_depth_loss.py -v`
Expected: FAIL — `ImportError: cannot import name 'edge_aware_logl1_loss'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/loss_utils.py  (append)
def edge_aware_logl1_loss(pred, gt, rgb, mask):
    """dn-splatter's EdgeAwareLogL1.

    log() keeps a handful of grossly-wrong pixels (windows, reflections) from
    dominating. The exp(-|grad rgb|) weight down-weights the loss at image
    edges, which is where this completed depth bleeds across silhouettes.
    """
    logl1 = torch.log(1.0 + torch.abs(pred - gt))

    grad_x = torch.abs(rgb[:, :, :-1] - rgb[:, :, 1:]).mean(0, keepdim=True)
    grad_y = torch.abs(rgb[:, :-1, :] - rgb[:, 1:, :]).mean(0, keepdim=True)
    wx = torch.exp(-grad_x)
    wy = torch.exp(-grad_y)

    mx, my = mask[:, :, :-1], mask[:, :-1, :]
    lx = (logl1[:, :, :-1] * wx)[mx]
    ly = (logl1[:, :-1, :] * wy)[my]

    n = lx.numel() + ly.numel()
    if n == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return (lx.sum() + ly.sum()) / n
```

In `gaussian_renderer/__init__.py`, unpack the 5-tuple and add `"depth": depth, "alpha": alpha` to the returned dict. In `train.py` after line 103:

```python
if opt.lambda_depth > 0 and viewpoint_cam.sensor_depth is not None \
        and iteration >= opt.depth_from_iter:
    Ldepth = edge_aware_logl1_loss(
        render_pkg["depth"], viewpoint_cam.sensor_depth.float(),
        gt_image, viewpoint_cam.depth_mask)
    loss = loss + opt.lambda_depth * Ldepth
```

Add the fallback variants the spec lists, dispatched by name:

```python
# utils/loss_utils.py  (append, after edge_aware_logl1_loss)
def depth_loss_fn(name):
    """Resolve --depth_loss to a callable with signature (pred, gt, rgb, mask)."""
    def _masked(f):
        def g(pred, gt, rgb, mask):
            if mask.sum() == 0:
                return torch.zeros((), device=pred.device, dtype=pred.dtype)
            return f(pred[mask], gt[mask])
        return g

    table = {
        "edgeaware_logl1": edge_aware_logl1_loss,
        "logl1": _masked(lambda p, g: torch.log(1.0 + torch.abs(p - g)).mean()),
        "l1": _masked(lambda p, g: torch.abs(p - g).mean()),
        "huber": _masked(lambda p, g: torch.nn.functional.smooth_l1_loss(p, g)),
    }
    if name not in table:
        raise ValueError("unknown --depth_loss {!r}; choose from {}".format(
            name, sorted(table)))
    return table[name]
```

Add to `arguments/__init__.py`: `ModelParams` gets `self.depths = ""`, `self.depth_scale_file = ""`, `self.depth_max = 30.0`, `self.depth_on_cpu = False`; `OptimizationParams` gets `self.lambda_depth = 0.2`, `self.depth_from_iter = 0`, `self.depth_loss = "edgeaware_logl1"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/ -v`
Expected: all pass

Then a short smoke run to confirm training is stable:
`conda run -n fastgs python train.py -s <input3> -m /tmp/smoke --iterations 500 --depths 1`
Expected: loss decreases, no NaN. **A NaN or an exploding loss here almost certainly means the scale was not applied** — re-check `depth_scale.json` was found.

- [ ] **Step 5: Commit**

```bash
git add utils/loss_utils.py gaussian_renderer/__init__.py train.py arguments/__init__.py tests/test_depth_loss.py
git commit -m "feat: add EdgeAwareLogL1 depth loss to training

Matches dn-splatter's tuned preset (lambda 0.2) so the fast and slow
engines stay comparable and the tuning transfers."
```

---

### Task 8: Evaluation

PSNR is precisely the metric that cannot see this problem, so depth error on held-out views is the headline number.

**Files:**
- Create: `scripts/eval_depth.py`
- Create: `tests/test_eval_depth.py`

**Interfaces:**
- Produces: `depth_metrics(pred, gt, mask, scale_mm_per_unit) -> dict` with `mae_m`, `rmse_m`, `n_valid`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_depth.py
import torch
from scripts.eval_depth import depth_metrics


def test_perfect_prediction_is_zero_error():
    d = torch.rand(1, 8, 8) + 1.0
    m = torch.ones(1, 8, 8, dtype=torch.bool)
    r = depth_metrics(d, d, m, 4000.0)
    assert r["mae_m"] < 1e-6 and r["rmse_m"] < 1e-6


def test_errors_are_reported_in_metres():
    gt = torch.ones(1, 4, 4)
    pred = gt + 0.25          # 0.25 COLMAP units
    m = torch.ones(1, 4, 4, dtype=torch.bool)
    r = depth_metrics(pred, gt, m, 4000.0)
    assert abs(r["mae_m"] - 1.0) < 1e-4      # 0.25 * 4000mm = 1.0 m


def test_only_valid_pixels_count():
    gt = torch.ones(1, 4, 4)
    pred = gt.clone()
    pred[0, 0, 0] = 50.0
    m = torch.ones(1, 4, 4, dtype=torch.bool)
    m[0, 0, 0] = False
    r = depth_metrics(pred, gt, m, 4000.0)
    assert r["mae_m"] < 1e-6 and r["n_valid"] == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/test_eval_depth.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eval_depth.py
import torch


def depth_metrics(pred, gt, mask, scale_mm_per_unit):
    """Depth error in metres. Inputs are in COLMAP units."""
    to_m = float(scale_mm_per_unit) / 1000.0
    if mask.sum() == 0:
        return {"mae_m": float("nan"), "rmse_m": float("nan"), "n_valid": 0}
    err = (pred - gt)[mask] * to_m
    return {
        "mae_m": err.abs().mean().item(),
        "rmse_m": err.pow(2).mean().sqrt().item(),
        "n_valid": int(mask.sum().item()),
    }
```

Add a `__main__` that loads a trained model, renders every test camera, and prints mean MAE/RMSE plus the Gaussian count.

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 5: Commit and run the comparison**

```bash
git add scripts/eval_depth.py tests/test_eval_depth.py
git commit -m "feat: depth error metrics on held-out views"
```

Then the measurement that decides whether this worked, both with `--eval`:

| run | command |
|---|---|
| baseline | `train.py -s <scene> -m out/base --eval` |
| depth | `train.py -s <scene> -m out/depth --eval --depths 1 --init_from_depth` |

Record for each: depth MAE/RMSE (must improve substantially), PSNR/SSIM/LPIPS (must not regress meaningfully), Gaussian count (expected to fall), and wall-clock (should stay near upstream; ~10–15 % overhead is acceptable).

---

### Task 9: Metric PLY export

The spec offers `--rescale_to_metric` as an opt-in: training stays in COLMAP units so FastGS's `cameras_extent`-normalised schedule is untouched, but the exported PLY can be written in metres so it measures correctly downstream.

**Files:**
- Modify: `scene/gaussian_model.py` (`save_ply`)
- Modify: `scene/__init__.py` (`Scene.save`)
- Modify: `arguments/__init__.py` (`ModelParams`)
- Create: `tests/test_metric_export.py`

**Interfaces:**
- Consumes: `scale_mm_per_unit` from `depth_scale.json` (Task 2)
- Produces: `GaussianModel.save_ply(path, metric_scale=None)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metric_export.py
import numpy as np
from scene.gaussian_model import apply_metric_scale


def test_positions_and_scales_convert_to_metres():
    xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    log_scales = np.log(np.array([[0.5, 0.5, 0.5]], dtype=np.float32))
    xyz_m, log_scales_m = apply_metric_scale(xyz, log_scales, 4000.0)
    # 4000 mm/unit -> 4 m/unit
    assert np.allclose(xyz_m, xyz * 4.0)
    assert np.allclose(np.exp(log_scales_m), np.exp(log_scales) * 4.0, rtol=1e-5)


def test_identity_when_scale_is_one_metre_per_unit():
    xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    log_scales = np.zeros((1, 3), dtype=np.float32)
    xyz_m, log_scales_m = apply_metric_scale(xyz, log_scales, 1000.0)
    assert np.allclose(xyz_m, xyz)
    assert np.allclose(log_scales_m, log_scales)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/test_metric_export.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_metric_scale'`

- [ ] **Step 3: Write minimal implementation**

```python
# scene/gaussian_model.py  (module level)
def apply_metric_scale(xyz, log_scales, scale_mm_per_unit):
    """Convert positions and log-scales from COLMAP units to metres.

    Scales are stored logarithmically, so a multiplicative world scaling is an
    additive shift in log space. Rotations and opacities are scale-invariant
    and must NOT be touched.
    """
    import numpy as np
    m_per_unit = float(scale_mm_per_unit) / 1000.0
    return xyz * m_per_unit, log_scales + np.log(m_per_unit)
```

In `save_ply`, accept `metric_scale=None` and apply it to the `xyz` and `scale` arrays just before assembling the PLY elements. In `Scene.save`, pass `self.depth_scale` when `args.rescale_to_metric` is set. Add `self.rescale_to_metric = False` to `ModelParams`.

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec rnd-3dgs conda run -n fastgs python -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add scene/gaussian_model.py scene/__init__.py arguments/__init__.py tests/test_metric_export.py
git commit -m "feat: optional metric PLY export

Training stays in COLMAP units so cameras_extent-normalised schedules are
untouched; only the export converts. Log-scales shift additively."
```

---

## Deferred

Depth-guided densification and pruning — feeding depth consistency into `compute_gaussian_score_fastgs` — is held back deliberately so the loss can be measured against a clean baseline. It becomes a small increment once Task 8 produces numbers.
