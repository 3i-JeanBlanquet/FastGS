#
# Forward-pass coverage for the depth/alpha outputs of the FastGS rasterizer.
#
# The rasterizer accumulates D = sum_i(d_i * alpha_i * T_i) and reports
# alpha = 1 - T_final. For a scene holding a single Gaussian at view-space
# depth d the two collapse to D = d * alpha, so D / alpha recovers d exactly
# for every covered pixel regardless of how much opacity the splat carries.
# The tests below lean on that identity rather than on any particular splat
# footprint, which keeps them independent of the low-pass filter and of the
# compact-box tile culling.
#

import math

import numpy as np
import pytest
import torch

from utils.graphics_utils import getProjectionMatrix, getWorld2View2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FOV = math.pi / 2
ZNEAR = 0.01
ZFAR = 100.0


def _settings(W=64, H=64, fov=FOV):
    """Rasterizer settings for an identity camera at the origin looking down +z.

    The matrices are composed exactly the way scene/cameras.py composes them,
    so the projection convention matches the rest of the codebase.
    """
    from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings

    R = np.eye(3)
    T = np.zeros(3)
    world_view_transform = torch.tensor(
        getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(ZNEAR, ZFAR, fov, fov).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
        projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    tan = math.tan(fov * 0.5)
    return GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tan,
        tanfovy=tan,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        mult=0.5,
        prefiltered=False,
        debug=False,
        get_flag=False,
        metric_map=torch.zeros(H * W, dtype=torch.int, device="cuda"))


def _one_gaussian(z, opacity=0.99, scale=0.3, requires_grad=False):
    """Keyword arguments for rasterising a single white splat at (0, 0, z)."""
    def t(values, shape):
        out = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(shape)
        out.requires_grad_(requires_grad)
        return out

    return dict(
        means3D=t([0.0, 0.0, z], (1, 3)),
        means2D=torch.zeros((1, 4), dtype=torch.float32, device="cuda",
                            requires_grad=requires_grad),
        opacities=t([opacity], (1, 1)),
        scales=t([scale, scale, scale], (1, 3)),
        rotations=t([1.0, 0.0, 0.0, 0.0], (1, 4)),
        colors_precomp=t([1.0, 1.0, 1.0], (1, 3)))


def _render(**gaussian):
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    return GaussianRasterizer(_settings())(**gaussian)


def test_rasterizer_returns_colour_radii_counts_depth_and_alpha():
    out = _render(**_one_gaussian(z=3.0))
    assert len(out) == 5
    color, _radii, _counts, depth, alpha = out
    assert color.shape == (3, 64, 64)
    assert depth.shape == (64, 64)
    assert alpha.shape == (64, 64)


def test_single_opaque_gaussian_renders_its_own_depth():
    _color, _radii, _counts, depth, alpha = _render(**_one_gaussian(z=3.0))
    covered = alpha > 0.5
    assert covered.sum() > 0, "the splat must cover some pixels"
    # D = d * alpha for a lone splat, so the alpha-normalised depth is exact.
    recovered = depth[covered] / alpha[covered]
    assert torch.allclose(recovered, torch.full_like(recovered, 3.0), atol=0.01)


def test_raw_depth_approximates_distance_where_the_splat_is_opaque():
    _color, _radii, _counts, depth, alpha = _render(**_one_gaussian(z=3.0))
    # Opacity is clamped to 0.99 in the kernel, so raw D tops out at 0.99 * d.
    opaque = alpha > 0.95
    assert opaque.sum() > 0, "the splat must saturate some pixels"
    assert torch.allclose(depth[opaque], torch.full_like(depth[opaque], 3.0), atol=0.15)


def test_depth_tracks_distance():
    d_near = _render(**_one_gaussian(z=2.0))[3]
    d_far = _render(**_one_gaussian(z=6.0))[3]
    assert d_far.max() > d_near.max() + 1.0


def test_alpha_is_zero_where_nothing_is_drawn():
    alpha = _render(**_one_gaussian(z=3.0))[4]
    assert alpha.min() < 0.01, "empty pixels must have zero coverage"
    assert alpha.max() > 0.5, "the splat must cover some pixels"


def test_depth_is_zero_where_nothing_is_drawn():
    _color, _radii, _counts, depth, alpha = _render(**_one_gaussian(z=3.0))
    empty = alpha < 1e-6
    assert empty.sum() > 0
    assert depth[empty].abs().max() == 0.0


def test_nearer_gaussian_dominates_the_blended_depth():
    near = _one_gaussian(z=2.0)
    far = _one_gaussian(z=8.0)
    both = dict(
        means3D=torch.cat([near["means3D"], far["means3D"]]),
        means2D=torch.zeros((2, 4), dtype=torch.float32, device="cuda"),
        opacities=torch.cat([near["opacities"], far["opacities"]]),
        scales=torch.cat([near["scales"], far["scales"]]),
        rotations=torch.cat([near["rotations"], far["rotations"]]),
        colors_precomp=torch.cat([near["colors_precomp"], far["colors_precomp"]]))

    _color, _radii, _counts, depth, alpha = _render(**both)
    centre = depth.shape[0] // 2
    assert alpha[centre, centre] > 0.5, "the splats must cover the centre pixel"
    # The near splat is nearly opaque, so it swallows almost all of the
    # transmittance and the blended depth stays close to its own.
    assert depth[centre, centre] / alpha[centre, centre] < 3.0


# --------------------------------------------------------------------------
# Multi-bucket regression guard for the sampled_ar checkpoint stride.
#
# The forward pass checkpoints its accumulators every 32 Gaussians into
# sampled_ar, and the backward pass replays them from
# `global_bucket_idx * BLOCK_SIZE * (CHANNELS + 1)`. Adding depth widened that
# stride from CHANNELS to CHANNELS + 1, and a mismatch between the writer and
# the reader silently corrupts the replayed colour accumulator.
#
# Two properties are needed to exercise it at all, and the single-splat tests
# above have neither:
#   * more than 32 Gaussians in ONE tile, so the bucket index reaches >= 1 --
#     at bucket 0 the stride term is multiplied by zero and cannot be wrong;
#   * a BACKWARD pass, because sampled_ar is write-only going forward
#     (out_color is written from the register C[ch], never read back from the
#     checkpoint buffer).
#
# dL_dcolors does not depend on sampled_ar, so dL_dopacity is the diagnostic
# gradient: the replayed `ar` accumulator feeds dL_dalpha and hence opacity.
# This is meaningful today -- the COLOUR backward already replays through
# sampled_ar, so a stride regression corrupts colour gradients right now,
# independently of the depth gradients Task 6 will add.
# --------------------------------------------------------------------------

TILE = 16       # BLOCK_X == BLOCK_Y == 16, so a 16x16 image is exactly one tile
CROWD = 150     # >32 splats in that one tile -> ceil(150/32) = 5 buckets


def _crowded_scene(requires_grad=False):
    """150 overlapping splats inside a single 16x16 tile. Deterministic."""
    g = torch.Generator(device="cpu").manual_seed(0)
    xy = (torch.rand((CROWD, 2), generator=g) - 0.5) * 0.8
    z = 2.0 + torch.rand((CROWD, 1), generator=g) * 2.0
    return dict(
        means3D=torch.cat([xy, z], dim=1).cuda(),
        means2D=torch.zeros((CROWD, 4), device="cuda", requires_grad=True),
        opacities=(0.2 + 0.5 * torch.rand((CROWD, 1), generator=g)).cuda()
                  .requires_grad_(requires_grad),
        scales=torch.full((CROWD, 3), 0.25, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(CROWD, 1).cuda(),
        colors_precomp=torch.rand((CROWD, 3), generator=g).cuda())


def _render_crowd(scene):
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    return GaussianRasterizer(_settings(W=TILE, H=TILE))(**scene)


def _colour_loss(scene, target):
    # Reduce in float64: a float32 sum over the image loses enough precision
    # that a central difference suffers catastrophic cancellation.
    color = _render_crowd(scene)[0]
    return ((color.double() - target.double()) ** 2).sum()


def test_crowded_scene_spans_more_than_one_bucket():
    scene = _crowded_scene()
    _color, radii, _counts, _depth, alpha = _render_crowd(scene)
    # The image is a single tile, so every visible splat lands in that tile.
    # More than 32 of them means the forward checkpoint runs with bbm >= 1.
    assert (radii > 0).sum() > 32, "need >32 splats in one tile for >1 bucket"
    assert (alpha > 0.5).sum() > 0, "the splats must actually cover pixels"


def test_opacity_gradients_survive_the_multi_bucket_replay():
    scene = _crowded_scene(requires_grad=True)
    g = torch.Generator(device="cpu").manual_seed(7)
    target = torch.rand((3, TILE, TILE), generator=g).cuda()

    _colour_loss(scene, target).backward()
    analytic = scene["opacities"].grad.clone().squeeze(1)

    eps = 5e-3
    worst = 0.0
    checked = 0
    for i in range(0, CROWD, 11):
        numeric = []
        for sign in (+1.0, -1.0):
            perturbed = _crowded_scene()
            with torch.no_grad():
                perturbed["opacities"][i, 0] += sign * eps
            numeric.append(_colour_loss(perturbed, target).item())
        finite_difference = (numeric[0] - numeric[1]) / (2 * eps)
        # Floor the scale so near-zero gradients, where the float32 finite
        # difference is pure noise, cannot dominate the comparison.
        scale = max(0.05, abs(analytic[i].item()), abs(finite_difference))
        worst = max(worst, abs(analytic[i].item() - finite_difference) / scale)
        checked += 1

    assert checked > 5
    # A stride mismatch corrupts the replayed accumulator outright and drives
    # this to ~1.0 (100% error); the true stride sits at the ~0.03 noise floor.
    assert worst < 0.15, "worst relative gradient error {:.4f}".format(worst)
