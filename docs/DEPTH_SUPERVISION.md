# FastGS with Metric Depth Supervision

A fork of [FastGS](https://github.com/fastgs/FastGS) that supervises training with metric
sensor depth, so the exported PLY has correct geometry instead of floaters and
ray-smeared splats.

## Why

RGB-only supervision is depth-ambiguous. A Gaussian placed anywhere along a view ray
produces the same pixel, so photometric loss cannot tell a splat on a wall from a splat
floating in front of it. The usual symptoms — floaters, fuzzy surfaces, splats smeared
along view rays — are not tuning failures. They are the loss function doing exactly what
it was asked to do.

Upstream FastGS supervises on `L1 + DSSIM` only, and its rasterizer emits colour and radii
and nothing else. This fork makes the rasterizer emit depth, backpropagate through it, and
adds a depth term to the loss.

## The constraint that shapes everything

**COLMAP's world is not metric.** Measured on two captures from the same rig:

| capture | scale |
|---|---|
| `rnd.project.mesh/test/input3` | 4135.3 mm per COLMAP unit |
| `rnd.project.reconstruction/test` | 3939.7 mm per COLMAP unit |

Both cluster near 4 m/unit, but they differ by 5.0%, and within a single scene the
per-image spread is ±13%. Depth PNGs are millimetres. Supervising millimetres directly
against a world where one unit is ~4 metres puts a ~4000× mismatch into the gradient.
That is not a tuning problem — it diverges on iteration one.

So the scale must be **measured per scene**. It cannot be hardcoded, and it cannot be
approximated at 4.0.

A second property was established the same way: the depth maps store **planar Z**
(distance along the optical axis), not radial distance. On these 90° cube faces the two
differ by up to 1.73× at the corners, so getting it wrong would bend every wall.

| convention | spread of ratio (MAD) | residual corr. with radial factor |
|---|---|---|
| **Planar Z** | 2.80% / 8.28% | +0.192 / **+0.025** |
| Radial | 10.07% / 11.01% | −0.247 / −0.464 |

(Two scenes shown. The near-zero residual correlation on the second is the exact signature
expected when the convention is right.) This is now asserted automatically per scene rather
than configured, so it cannot be set wrong.

## Architecture

Three stages.

### 1. Calibrate — `scripts/estimate_depth_scale.py`

CPU only, runs once per scene, no torch or CUDA required. Projects the COLMAP sparse points
into every image, samples the depth PNG at each keypoint, and takes the robust median ratio.
Rejects per-image outliers by MAD, asserts the planar convention, and writes
`depth_scale.json` into the model directory. Fails loudly rather than guessing.

```
$ python scripts/estimate_depth_scale.py -s /path/to/scene
scale      : 4135.3 mm per COLMAP unit
convention : planar (margin 1.33x)
samples    : 32723 over 102 images
outliers   : 10
```

Those outliers are worth reading. One image in the reference capture calibrates at
385 mm/unit against ~3940 — a 10× error, meaning a bad pose or bad depth. If it is the
pose, it is degrading RGB-only training too.

### 2. Rasterize — CUDA

The forward pass accumulates one extra channel alongside colour and exposes the
already-computed transmittance as alpha:

```
D = Σ dᵢ·αᵢ·Tᵢ        alpha = 1 − T_final
```

The backward pass routes `dL/dD` to **two** places, and both matter:

- **→ each Gaussian's view-space z**, which slides splats along the ray onto the surface.
- **→ alpha** (opacity, scale, rotation), which is what actually *removes* floaters: a splat
  hovering in front of a wall makes rendered depth too small, so its opacity gets pushed down.

An implementation with only the first path passes a naive smoke test and completely fails
the feature's purpose.

### 3. Supervise — Python

Loader reads `<stem>_depth.png` beside each image (or a parallel `depths/` directory),
masks in millimetres before converting to COLMAP units, and resizes with NEAREST only —
interpolating across a depth discontinuity invents surfaces at neither depth.

The loss matches dn-splatter's tuned preset, so the fast and slow engines stay comparable:

```
L_depth = mean( log(1 + |D − D_gt|) · exp(−|∇RGB|) )   over masked pixels
loss    = (1−λ_dssim)·L1 + λ_dssim·(1−SSIM) + λ_depth·L_depth
```

The edge-aware weight down-weights depth loss where the RGB image has strong gradients —
exactly where object silhouettes are, and exactly where completed depth bleeds. The log
compresses large residuals so a handful of grossly wrong pixels (windows, reflections)
cannot dominate.

**Not inverse depth**, deliberately. INRIA's implementation renders inverse depth because
monocular predictors emit it; on bounded metric depth an inverse-depth L1 would weight a
chair at 1 m about 64× more than a wall at 8 m — backwards, since walls and floors are
where the floaters live.

## Does it work?

Held-out evaluation on the reference capture (112 images, 30000 iterations, `--eval` holds
out every 8th camera; 14 test views, 12 contributing valid pixels in every run):

| model | splats | depth MAE | depth RMSE | PSNR | SSIM | mask coverage |
|---|---:|---:|---:|---:|---:|---:|
| baseline (RGB only) | 101,410 | **2.52 m** | 3.25 m | 23.43 | 0.868 | 88.8% |
| `--depths` (sparse init) | 102,003 | **0.72 m** | 1.34 m | 23.56 | 0.872 | 86.7% |
| `--depths --init_from_depth` | 280,205 | **0.72 m** | 1.33 m | 23.23 | 0.860 | 88.8% |

**Depth error falls 3.5× while PSNR moves 0.13 dB.** That gap is the entire argument for this
work: photometric metrics cannot see the geometry problem, so a scene full of floaters scores
well on PSNR. Measure depth error, or you are not measuring what you set out to fix.

Coverage is comparable across all three (86.7–88.8% of pixels), so the comparison is like for
like rather than one model being scored only where it is already confident.

### Recommended configuration: `--depths` alone

Dense initialisation buys **no additional depth accuracy** (0.7178 vs 0.7168 m MAE — a 1 mm
difference) while costing 2.7× the splats, 2.8× the PLY size, and slightly *worse* PSNR and
SSIM. Use `--init_from_depth` only if you specifically want a denser cloud for another reason.

This contradicts the original design's expectation on two counts, both worth recording: depth
supervision did not reduce Gaussian count (it was flat), and dense initialisation — predicted
to be independently valuable — turned out to be the expensive half with no measured geometric
benefit on this capture.

## Performance

Measured on an RTX 6000 Ada, 500k Gaussians at 1024×1024, 12 interleaved repetitions per
configuration. Interleaving matters: a first naive run measured the modified build as
*faster* than baseline, an artifact of running configurations sequentially while GPU clocks
ramped.

| configuration | median ms/iter | peak MB |
|---|---|---|
| upstream baseline | 1.667 | 895.5 |
| this fork, depth **off** | 1.794 (+7.6%) | 1024.3 (+14.4%) |
| this fork, depth **on** | 1.820 (+9.1%) | 1026.0 (+14.6%) |

**The cost is almost entirely unconditional.** Enabling depth adds only ~1.5 percentage
points over merely having the modified kernels. The overhead comes from widening the
per-bucket checkpoint buffer `sampled_ar` from `CHANNELS` to `CHANNELS+1`, which every user
pays whether or not they supervise on depth.

Read the memory figures as solid and the timings as approximate. Memory is deterministic.
The timing distribution is noisy (pooled σ ≈ 0.70 ms) because the benchmark box also runs
production containers — the mean sits far above the median, the signature of contention.
Medians are quoted for that reason.

### End-to-end, which is the number that actually matters

The microbenchmark above measures the rasterizer in isolation and **understates real cost**.
Full 30000-iteration runs on the reference capture (112 images at 1024², RTX 6000 Ada):

Measured back to back on the same box under identical load, with both optimisations below
applied:

| configuration | splats | training time | vs baseline |
|---|---|---|---|
| baseline (RGB only) | 100,274 | 114.98 s | — |
| `--depths` | 100,772 | 148.87 s | **+29.5%** |

Absolute times vary with what else is running on that machine; run baseline and depth
back to back and compare the ratio, never absolute times from different sessions.

Two optimisations took this from +65% to +29.5%, and the route to them is worth recording
because the first attempt was based on a wrong inference:

**Attribution by subtraction is not profiling.** Subtracting a rasterizer microbenchmark from
the end-to-end delta suggested ~91% of the cost was recomputing the edge weights each
iteration. Caching them (they depend only on the static ground-truth image) recovered just
3.7%. Direct profiling of the loss found the real culprit: **boolean-mask advanced indexing**,
`logl1[mask]`, at 0.256 ms/iter — half the entire loss. PyTorch must learn the output size
before allocating, forcing a device→host sync every call. Rewriting as multiply-and-normalise
is bit-identical (relative difference 0.00e+00) and drops the loss from 0.513 to 0.151 ms.

| loss component (1024²) | ms/iter |
|---|---|
| fp16 → float cast | 0.013 |
| alpha mask construction | 0.029 |
| `log1p(|pred−gt|)` | 0.027 |
| **boolean-mask index** | **0.256** |
| same via multiply | 0.043 |

Dense initialisation remains the expensive option: it nearly triples splat count
(101k → 280k) and PLY size (25 MB → 69 MB) for no measured depth-accuracy gain.

**Dense initialisation is the expensive part, and it changes the output.** It nearly triples
the splat count (101k → 280k) and the PLY size (25 MB → 69 MB). Note this contradicts the
original design's expectation that depth supervision would *reduce* Gaussian count — with
sparse init the count is flat, and with dense init it rises sharply, because the count is
dominated by initialisation rather than by the loss. If PLY weight matters more to you than
initial geometry, run with `--depths` alone.

Dense init does pay for itself in one respect: the surviving supervision mask is **96.2%**
with it versus **10.7%** without at the same early iteration, because dense coverage raises
alpha immediately. The two features reinforce each other.

## Usage

```bash
# 1. calibrate once per scene (CPU, seconds)
python scripts/estimate_depth_scale.py -s /path/to/scene

# 2. train
python train.py -s /path/to/scene -m /path/to/output \
    --depths 1 --init_from_depth --eval
```

| flag | default | purpose |
|---|---|---|
| `--depths` | `""` | enable depth supervision (any non-empty value) |
| `--depth_scale_file` | `<model>/depth_scale.json` | calibration from stage 1 |
| `--lambda_depth` | `0.2` | depth loss weight (matches dn-splatter) |
| `--depth_loss` | `edgeaware_logl1` | `edgeaware_logl1` \| `logl1` \| `l1` \| `huber` |
| `--depth_max` | `30.0` | metres; reject beyond |
| `--depth_from_iter` | `0` | delay depth supervision |
| `--init_from_depth` | `False` | dense backprojected init instead of COLMAP sparse |
| `--depth_init_voxel` | `0.02` | voxel size for that init |
| `--depth_on_cpu` | `False` | keep depth off-GPU on smaller cards |

### Expected input layouts

Both work, detected automatically:

```
scene/                          scene/
  0/            <- COLMAP         sparse/0/      <- COLMAP
  rig-1,0,0,0/                    images/
    <uuid>.jpg                    depths/
    <uuid>_depth.png
```

Depth is uint16 PNG in millimetres, `0 = invalid`, same resolution as its image.

## Known limitations and open questions

**The alpha threshold is settled, but takes time to earn coverage with sparse init.** Rendered
depth is *unnormalised* — where coverage is partial it under-reports distance, so the loss
masks on `alpha > 0.95`. With `--init_from_depth` coverage is 96% from the first iteration.
With COLMAP sparse init it starts at 10.7%, but climbs through training to 0.70–0.99
(averaging ~0.9) and reaches 86.7% mean coverage on held-out views by 30k. Both paths end up
supervising nearly the whole image; sparse init simply gets there later. The training loop
logs this fraction every 1000 iterations so a genuinely degenerate mask stays visible.

**Depth-guided densification and pruning is deliberately not implemented.** Feeding depth
consistency into FastGS's `compute_gaussian_score_fastgs` is the right eventual answer for
floater removal, but landing it alongside the loss would make results unattributable.

**Surface extraction is out of scope.** Depth–normal consistency and scale flattening
(2DGS/GOF-style) target mesh-extractable surfaces, a different goal that costs PSNR.

**Sensor depth here is completed/learned, not raw.** Object silhouettes bleed several
pixels. The edge-aware loss weight is the mitigation, not a cure.

## Notes for whoever builds this next

The `fastgs` conda env's torch is built against **CUDA 11.6**, but the container sets
`CUDA_HOME=/usr/local/cuda-12.1` globally, and torch's extension builder hard-errors on the
mismatch. Build with:

```bash
export CUDA_HOME=/opt/conda/envs/fastgs      # the env prefix IS a complete 11.6 toolkit
export TORCH_CUDA_ARCH_LIST="8.6+PTX"        # 11.6 cannot target sm_89; Ada runs the PTX JIT
python setup.py build_ext --inplace
```

Prefer `build_ext --inplace` plus `PYTHONPATH` shadowing over `pip install -e .` when the
target env is one a production pipeline runs — installing replaces the deployed rasterizer.

A clean `compute-sanitizer` run is evidence about one allocation layout, not proof of
in-bounds arithmetic. Run it with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, or an out-of-bounds
read simply lands in a neighbouring allocation and reports clean.
