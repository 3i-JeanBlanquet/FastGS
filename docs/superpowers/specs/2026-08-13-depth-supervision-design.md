# Depth Supervision for FastGS

**Status:** design, awaiting review
**Date:** 2026-08-13
**Fork of:** [fastgs/FastGS](https://github.com/fastgs/FastGS) (MIT)

## Problem

RGB-only supervision is depth-ambiguous. A Gaussian placed anywhere along a view
ray produces the same pixel, so photometric loss cannot distinguish a splat on a
wall from a splat floating in front of it. The result in practice is floaters,
fuzzy surfaces, and splats smeared along view rays — visible as poor geometry in
the exported PLY even when PSNR looks healthy.

Upstream FastGS supervises on `L1 + DSSIM` only (`train.py:101-103`). Its
rasterizer emits colour and radii and nothing else — no depth, no alpha
(`gaussian_renderer/__init__.py:106`). Cameras carry no depth field
(`scene/cameras.py`), and the dataset readers never look for depth
(`scene/dataset_readers.py`).

This design adds metric depth supervision so the geometry is constrained
directly.

## Pipeline context

FastGS is already deployed inside `rnd.gaussian_splatting` as the fast half of a
hybrid engine (`run.py:325 run_fastgs`), alongside **dn-splatter**, which is
itself a depth- and normal-supervised splatting method. The deployed FastGS copy
(commit `fe3a518`) is unmodified vanilla — `train.py` contains no depth
references.

So the goal is not to invent depth supervision for this stack. It is to give the
*fast* engine the capability the *slow* engine already has, so choosing FastGS
stops meaning giving up geometry.

That has a direct consequence for this design: **match dn-splatter's depth loss
rather than inventing one.** Its settings are already tuned on this exact
capture (`run.py:37 DEPTH_PRESET`):

```python
use_depth_loss=True, depth_loss_type="EdgeAwareLogL1",
depth_lambda=0.2, normal_supervision="depth"
```

Matching means the two engines are comparable, and the tuning knowledge
transfers instead of being rediscovered.

### What the existing stack independently confirms

`dn_splatter/data/livo_dataparser.py` corroborates both measured facts below,
from a completely different direction:

| dataparser setting | confirms |
|---|---|
| `depth_unit_scale_factor = 0.001` | depth PNGs are millimetres |
| `is_euclidean_depth = False` | depth is **z-distance, not Euclidean** — matches the 2.80 % vs 10.07 % measurement |

### Resolved: FastGS receives non-metric COLMAP poses, at a per-scene scale

`auto_scale_poses = False` describes the dn-splatter/LIVO path, not the FastGS
one. Measured on the canonical FastGS input (`rnd.project.mesh/test/input3`,
28 stations × 4 faces) and cross-checked against a second capture:

| scene | scale (mm/unit) | planar MAD | radial corr | camera bbox (m) |
|---|---|---|---|---|
| `rnd.project.reconstruction/test` | 3939.7 | 2.80 % | −0.247 | 37.1 × 0.42 × 24.3 |
| `rnd.project.mesh/test/input3` | **4135.3** | 8.28 % | −0.464 | 25.6 × 1.11 × 11.1 |

Both cluster near 4 m/unit, which suggests a systematic normalisation rather
than a freely-drifting SfM scale. But they differ by **5.0 %**, and scene 2's
per-image spread is ±13.3 %. **The scale must therefore be measured per scene.**
It cannot be hardcoded, and it cannot be approximated at 4.0.

This makes Stage 1 mandatory rather than conditional.

Planar Z is confirmed a second time and more strongly: scene 2's residual
correlation with the radial factor is **+0.025** — essentially zero, the exact
signature expected when the convention is right — against radial's −0.464.

## Target data

Measured from the reference capture at
`rnd.toolkit/pipeline/rnd.project.reconstruction/test`:

| property | value |
|---|---|
| Layout | rig folders at the input root, COLMAP model in `0/` |
| Rig | 4 pinhole cube faces per station, yaw 0°/90°/180°/270° |
| Stations | 44 (176 images); canonical `input3` sample is 28 (112 images) |
| Intrinsics | `fx = fy = 506.91`, `cx = cy = 512`, 1024×1024 → 90.6° FOV |
| RGB | `<uuid>.jpg`, 1024×1024 |
| Depth | `<uuid>_depth.png`, 1024×1024, uint16, **0 = invalid** |
| Invalid pixels | 0.07 % – 1.8 % per image |
| COLMAP model | 3.12 format (`rigs.bin`/`frames.bin` present, ignored by the loader) |

RGB and depth are the same resolution and sit side by side in the same rig
folder, so no resizing and no directory reorganisation are required.

## Measured facts that drive the design

Two properties were established empirically before any code was written, by
projecting the COLMAP sparse points into each image and comparing against the
depth PNG at the corresponding pixel (49,916 matched samples, 176 images).

### 1. Depth is planar Z, not radial distance

| convention | MAD of ratio | corr(residual, radial factor) |
|---|---|---|
| **Planar Z** | **2.80 %** | +0.192 |
| Radial | 10.07 % | −0.247 |

The radial correction factor `sqrt(1 + xn² + yn²)` reaches 1.709 at the corners
of these 90° faces, so the two hypotheses are separated by up to 71 % — far
above the noise floor. Planar Z agrees 3.6× more tightly, and the sign flip in
the correlation is the signature of over- versus under-correction.

The rasterizer natively produces planar Z, so **no conversion is required**.
This is asserted per scene rather than assumed (see Stage 1).

### 2. The COLMAP world is not metric — 1 unit ≈ 3.94 m

`median(depth_mm / z_colmap) = 3939.7 mm per COLMAP unit`, consistent across
images (p5–p95: 3692–4108, i.e. ±5 %). It is a single global scale, not drift.

Sanity check: camera centres span 37.1 m × **0.42 m** × 24.3 m once scaled — a
large office floor captured from a fixed-height tripod. The 0.42 m vertical
spread independently confirms the calibration.

**This is the critical constraint.** Supervising millimetres against a world
where one unit is 3.94 m injects a ~3900× mismatch straight into the gradient.
A metric depth loss applied naively does not train badly; it diverges
immediately. Any implementation that omits calibration is wrong regardless of
how well the rest is built.

One image reported a scale of 385 mm/unit — a 10× outlier — indicating a bad
pose or bad depth for that frame. Outlier images must be detected and excluded.

## Architecture

Three stages. Stage 1 is new and was not in the original two-stage sketch; it
exists because of finding 2.

```
Stage 1  CALIBRATE   (CPU, once per scene)
   COLMAP sparse points + depth PNGs -> depth_scale.json
   { scale, convention assertion, per-image outliers }

Stage 2  RASTERIZE   (CUDA)
   forward:  D = sum_i d_i * alpha_i * T_i ,  A = 1 - T_final
   backward: dL/dD -> per-Gaussian view-space z, and -> alpha

Stage 3  SUPERVISE   (Python)
   loader -> masked L1 in COLMAP units -> added to the RGB loss
   plus dense depth backprojection for initialisation
```

### Stage 1 — Calibration

**New file:** `scripts/estimate_depth_scale.py`

Productionises the probe used to establish the facts above.

- Reads `points3D.bin` / `images.bin` / `cameras.bin`, projects each sparse
  point into every image observing it, samples the depth PNG at that pixel.
- Robust global scale = median ratio across all samples.
- Per-image median ratio; images deviating by more than a configurable MAD
  multiple are written to an `outliers` list and excluded from training.
- Recomputes both convention hypotheses and **asserts** planar wins by a clear
  margin. If radial wins, or the margin is ambiguous, the script fails loudly
  rather than emitting a file. The `--depth_mode` flag discussed earlier is
  therefore replaced by an automatic, per-scene check.
- Emits `depth_scale.json` next to the COLMAP model:

```json
{
  "scale_mm_per_unit": 3939.7,
  "convention": "planar",
  "convention_margin": 3.6,
  "n_samples": 49916,
  "per_image": {"rig-1,0,0,0/027ba525-....jpg": 3944.1},
  "outliers": ["rig-0.7071.../<uuid>.jpg"]
}
```

**Decision: convert depth into COLMAP units rather than rescaling the world.**
3DGS normalises position learning rates and densification thresholds by
`cameras_extent`, so rescaling the world is *nearly* neutral — but "nearly" is
not a property worth betting FastGS's tuned 100-second schedule on. Dividing GT
depth by the scale is mathematically equivalent for the loss and touches
nothing else. Metric rescaling of the output PLY becomes an opt-in flag
(`--rescale_to_metric`) for the separate goal of a measurable PLY.

### Stage 2 — CUDA rasterizer

**Files:** `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/{forward,backward}.cu`,
`rasterizer_impl.cu`, `rasterize_points.cu`, `ext.cpp`, and the Python binding.

FastGS's rasterizer is taming-3dgs-derived: the backward pass replays forward
state from per-32-Gaussian buckets (`sampled_T`, `sampled_ar`). INRIA's current
`diff-gaussian-rasterization` shares that bucket architecture *and* already
renders depth, so this is a port against a proven reference rather than a
derivation from scratch.

**Forward** (`renderCUDA`, `forward.cu:275`):
- Accept the existing per-Gaussian `depths` array (already computed in
  preprocess for sorting) as an input to the render kernel.
- Accumulate `D += depths[id] * alpha * T` alongside the colour channels.
- Widen the per-bucket checkpoint `sampled_ar` from `CHANNELS` to `CHANNELS + 1`
  so the backward pass can replay the depth accumulator. This requires matching
  changes to the buffer sizing in `rasterizer_impl.cu`.
- Write `out_depth[pix_id] = D` and `out_alpha[pix_id] = 1 - T`. `final_T` is
  already computed (`forward.cu:422`); alpha is essentially free.

**Backward** (`backward.cu`):
- New input `dL_dout_depth`.
- Mirror the colour derivation with the depth channel, background zero:
  - `dL_dalpha += (d_i * T - accum_rec_depth) * dL_dD`
  - `dL_ddepths[id] += alpha * T * dL_dD` (atomic)
- In backward preprocess, route depth gradient to position. Since
  `z_view = V[2]·x + V[6]·y + V[10]·z + V[14]`:
  - `dL_dmean3D += dL_ddepth * (V[2], V[6], V[10])`

**Both gradient paths are required and serve different purposes.** The position
path slides Gaussians along the ray onto the true surface. The alpha path is
what actually *removes* floaters: a splat hovering in front of a wall makes
rendered depth too small, and the gradient pushes its opacity down. Implementing
only the position path would leave the headline problem unfixed.

**Alpha is returned without gradient**, for masking and diagnostics only. Indoors
`A ≈ 1` almost everywhere, so unnormalised `D` is unbiased where it matters, and
pixels with low alpha are masked out instead of normalised. This avoids a second
backward path for no measurable benefit.

### Stage 3 — Data, loss, initialisation

**`scene/dataset_readers.py`** — for each image, look for `<stem>_depth.png`
beside it; fall back to `image_mapper.json` if present. Missing depth is not an
error; those cameras train on RGB alone. Images in the calibration `outliers`
list are skipped.

**`utils/camera_utils.py`** — build the valid mask **while still in millimetres**,
then convert to COLMAP units (`d_png / scale_mm_per_unit`). Masking first keeps
every threshold expressed in physical units, so the flags stay meaningful
independently of whatever scale a given COLMAP run happens to produce:

1. `d > 0` (sensor invalid marker)
2. `d < --depth_max` (default 30 m; observed maxima reach 41 m through windows,
   where the depth is unreliable)
Silhouette bleed is handled by the loss rather than the mask — see below.

Storage: depth as fp16 plus a uint8 mask is ~553 MB across 176 cameras at 1024²,
on top of the ~2.2 GB the RGB images already occupy. Acceptable on 24 GB; a
`--depth_on_cpu` fallback is provided for smaller cards.

**`utils/loss_utils.py`** — port dn-splatter's `EdgeAwareLogL1` as the default:

```
L_depth = mean( log(1 + |D_render - D_gt|) * exp(-|grad(RGB)|) )[mask]
loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM) + lambda_depth * L_depth
```

Two properties matter here, and both are reasons to copy rather than invent.

The **edge-aware weight** `exp(-|grad(RGB)|)` down-weights the depth loss wherever
the RGB image has strong gradients — which is precisely where object silhouettes
are, and precisely where this completed/learned depth bleeds several pixels.
That is a more principled fix than the mask erosion originally proposed here: it
degrades smoothly with edge strength instead of making a hard include/exclude
decision, and it is already validated on this capture.

The **log** compresses large residuals, so the handful of grossly wrong pixels
(windows, reflective surfaces) cannot dominate the gradient.

`lambda_depth` defaults to **0.2**, matching `DEPTH_PRESET`, rather than a value
derived from first principles — it is a measured setting from the same data.
`--depth_loss {edgeaware_logl1, logl1, l1, huber}` allows falling back.

**Not inverse depth**, deliberately: INRIA renders inverse depth because
monocular predictors output it, but on bounded metric depth an inverse-depth L1
weights a chair at 1 m roughly 64× more than a wall at 8 m. Walls and floors are
where the floaters are, so that weighting is backwards for this goal. Rendering
true depth also keeps the parameterisation choice in Python, where it is cheap
to change.

**Initialisation** — `--init_from_depth` backprojects the depth maps into a
dense point cloud (voxel-downsampled) in place of COLMAP's 13,258 sparse points.
This is independently valuable and requires no CUDA, so it lands even if the
kernel work slips.

### CLI surface

| flag | default | purpose |
|---|---|---|
| `--depths` | `""` | enable depth supervision |
| `--depth_scale_file` | `<colmap>/depth_scale.json` | calibration from Stage 1 |
| `--lambda_depth` | `0.2` | depth loss weight (matches dn-splatter) |
| `--depth_loss` | `edgeaware_logl1` | `edgeaware_logl1` \| `logl1` \| `l1` \| `huber` |
| `--depth_max` | `30.0` | metres; reject beyond |
| `--depth_from_iter` | `0` | delay depth supervision |
| `--init_from_depth` | `False` | dense backprojected init |
| `--depth_on_cpu` | `False` | keep depth off-GPU |
| `--rescale_to_metric` | `False` | emit a metrically-scaled PLY |

## Build environment

Verified on `3i-instance-high` (RTX 6000 Ada, 48 GB, 32 cores), inside the
`rnd-3dgs` container's `fastgs` conda env:

| | |
|---|---|
| Python / torch | 3.7.13 / 1.12.1 + cu116 |
| nvcc | 11.6.55 — **present, so the rasterizer can be built here** |
| arch list | up to `sm_86`; the Ada card (`sm_89`) runs via PTX JIT |

CUDA 11.6 cannot target `sm_89` directly, so the extension builds `sm_86 + PTX`
as the existing Dockerfile already does. All CUDA work must compile under
**torch 1.12 / C++14 era APIs**, not the torch 2.x conventions used by current
upstream rasterizers — a constraint worth stating explicitly, since the INRIA
reference being ported from targets a newer toolchain.

Host disk is at 92 % (36 GB free of 457 GB). Build artifacts and a second set of
training outputs should be sized against that before long runs.

## Testing

The CUDA cannot be compiled on the development machine (macOS, no NVIDIA), but
it can be built and tested on the GPU node above, so verification is staged by
what can be checked where.

**Locally, today:**
- Calibration script against the reference capture; the planar/radial margin and
  the 3939.7 mm/unit scale are regression assertions with known-good values.
- Loader and mask construction on real files, including the invalid-pixel and
  silhouette-erosion behaviour.
- Backprojection: reprojecting one station's depth into a neighbouring station's
  camera must land within a few pixels.

**On the CUDA box:**
- **Finite-difference gradient check** on a small scene (~100 Gaussians, 32×32),
  perturbing positions and opacities and comparing analytic against numeric
  `dL/dD`. Kernels are fp32, so this is a relative-tolerance check (~1e-2), not
  `torch.autograd.gradcheck`.
- Analytic single-Gaussian test: one opaque Gaussian at known depth must render
  that depth.
- Alpha correctness: `A ≈ 1` on covered pixels, `≈ 0` on empty ones.

**End-to-end**, baseline versus depth-supervised on the 44-station scene:

| metric | expectation |
|---|---|
| Depth MAE/RMSE on held-out views | **primary** — must improve substantially |
| PSNR / SSIM / LPIPS | must not regress meaningfully |
| Gaussian count | expected to fall |
| Training time | must stay close to upstream; depth adds ~10–15 % |

Depth error on held-out views is the headline metric, since PSNR is precisely
the measure that cannot see this problem.

## Deliberately out of scope

**Depth-guided densification and pruning.** Feeding depth consistency into
FastGS's `compute_gaussian_score_fastgs` is the right eventual answer for floater
removal, but landing it at the same time as the loss makes the result
unattributable — a regression could come from either. It follows as a small
increment once there is a clean measured baseline.

**Surface extraction / mesh quality.** Depth–normal consistency and scale
flattening (2DGS/GOF-style) target mesh-extractable surfaces, which is a
different goal from the correct-geometry-and-fewer-floaters target here, and
costs PSNR.

**Pose refinement.** The calibration script will identify bad poses but will not
fix them.

## Risks

| risk | mitigation |
|---|---|
| Silhouette bleed in completed depth drags geometry into free space | `EdgeAwareLogL1`, already validated on this capture (Stage 3) |
| CUDA reference targets torch 2.x; build env is torch 1.12 / cu11.6 | port against 1.12-era APIs; build early to surface incompatibilities before the logic is finished |
| GPU host at 92 % disk | size artifacts before long runs |
| Unreliable depth through windows (up to 41 m observed) | `--depth_max` clamp; low-alpha masking |
| CUDA written without local compilation | port from a proven reference; finite-difference gradient check before any training run |
| `lambda_depth` mis-tuned, geometry fights photometry | `--depth_from_iter`; sweep on one scene before adopting |
| Scale calibration on a scene with too few sparse points | script reports sample counts and refuses below a threshold (46 of 176 images already fall below 20 matched points on the reference scene) |
