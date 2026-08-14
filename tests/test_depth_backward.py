#
# Backward-pass coverage for the depth output of the FastGS rasterizer.
#
# dL/dD has to reach two places and they do different jobs:
#
#   * every Gaussian's view-space z, via dL_ddepths -> dL_dmean3D. This slides
#     a splat along its view ray onto the true surface.
#   * alpha, via dL_dalpha -> opacity / scale / rotation. This is the path that
#     removes floaters: a splat hovering in front of a wall drags the rendered
#     depth too near, and the gradient pushes its opacity down until it goes.
#
# An implementation carrying only the first path renders a plausible-looking
# gradient and leaves the headline problem unfixed, so the two are covered
# separately below, each by a test that fails if only that path is removed.
#
# The rasterizer imports are function-local: the extension is only built on the
# GPU box, and a module-level import would break collection everywhere else.
#

import pytest
import torch

from tests.test_render_depth import CROWD, TILE, _crowded_scene, _one_gaussian, _render_crowd, _settings

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rendered_depth_sum(z_value, opacity=0.99):
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    g = _one_gaussian(z=z_value, opacity=opacity)
    g["means3D"] = g["means3D"].clone().requires_grad_(True)
    out = GaussianRasterizer(_settings())(**g)
    return out[3].sum(), g["means3D"]


# --------------------------------------------------------------------------
# Path 1: dL/dD -> per-Gaussian view-space depth -> 3D mean.
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Path 2: dL/dD -> alpha -> opacity, scale and rotation.
#
# Neither assertion below can be satisfied by the position path: dL_ddepths is
# folded straight into dL_dmeans3D and never reaches opacity or scale.
# --------------------------------------------------------------------------


def test_depth_gradient_flows_to_opacity():
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    g = _one_gaussian(z=3.0)
    g["opacities"] = g["opacities"].clone().requires_grad_(True)
    out = GaussianRasterizer(_settings())(**g)
    out[3].sum().backward()
    assert g["opacities"].grad is not None
    assert g["opacities"].grad.abs().sum() > 1e-6, \
        "the alpha path is what removes floaters -- it must carry gradient"


def test_depth_gradient_flows_to_scale():
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    g = _one_gaussian(z=3.0)
    g["scales"] = g["scales"].clone().requires_grad_(True)
    out = GaussianRasterizer(_settings())(**g)
    out[3].sum().backward()
    assert g["scales"].grad is not None
    # Shrinking the splat shrinks its footprint, which is only visible to the
    # depth loss through alpha.
    assert g["scales"].grad.abs().sum() > 1e-6


# --------------------------------------------------------------------------
# Multi-bucket coverage.
#
# The forward checkpoints its accumulators every 32 Gaussians into sampled_ar
# and the backward replays them from
# `global_bucket_idx * BLOCK_SIZE * (CHANNELS + 1)`, with the depth accumulator
# living in the extra slot at channel index CHANNELS. None of that arithmetic
# is exercised while the bucket index is 0, because the stride is multiplied by
# zero -- so a single-Gaussian test would leave the replay untested. The scenes
# below put well over 32 splats into ONE 16x16 tile so the index reaches >= 1.
#
# Both finite differences reduce in float64: a float32 sum over the image loses
# enough precision that a central difference suffers catastrophic cancellation.
# --------------------------------------------------------------------------


def _depth_loss(scene, target):
    depth = _render_crowd(scene)[3]
    return ((depth.double() - target.double()) ** 2).sum()


def _depth_target(seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (2.0 + 3.0 * torch.rand((TILE, TILE), generator=g)).cuda()


def test_depth_opacity_gradients_survive_the_multi_bucket_replay():
    """The alpha path, checked against finite differences across buckets.

    dL_dopacity is the diagnostic gradient for the depth replay just as it is
    for the colour replay: it is fed by the replayed depth accumulator ar[C],
    so a wrong stride or a wrong intra-bucket offset corrupts it outright.
    """
    scene = _crowded_scene(requires_grad=True)
    target = _depth_target()

    _depth_loss(scene, target).backward()
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
            numeric.append(_depth_loss(perturbed, target).item())
        finite_difference = (numeric[0] - numeric[1]) / (2 * eps)
        # Floor the scale so near-zero gradients, where the finite difference is
        # mostly quantisation noise from the alpha < 1/255 cutoff, cannot
        # dominate the comparison.
        scale = max(0.5, abs(analytic[i].item()), abs(finite_difference))
        worst = max(worst, abs(analytic[i].item() - finite_difference) / scale)
        checked += 1

    assert checked > 5
    assert worst < 0.15, "worst relative gradient error {:.4f}".format(worst)


Z_STEP = 0.02       # depth gap between neighbouring splats
Z_EPS = 2e-3        # central-difference step, an order of magnitude smaller


def _depth_ladder_scene(requires_grad=False):
    """CROWD splats in one 16x16 tile at strictly separated depths.

    Two properties matter and the random scene above has neither:

      * the depths are spaced by Z_STEP, well clear of Z_EPS. Perturbing one z
        must not reorder the depth-sorted splat list, because alpha blending is
        order dependent and a swap moves the loss discontinuously, which makes
        a central difference meaningless.
      * the opacities are low, so transmittance does not collapse after the
        first couple of dozen splats. The backward skips whole buckets past
        max_contrib, and a scene that saturates early would never reach the
        later buckets at all.
    """
    g = torch.Generator(device="cpu").manual_seed(3)
    xy = (torch.rand((CROWD, 2), generator=g) - 0.5) * 0.6
    z = 2.0 + Z_STEP * torch.arange(CROWD, dtype=torch.float32).unsqueeze(1)
    return dict(
        means3D=torch.cat([xy, z], dim=1).cuda().requires_grad_(requires_grad),
        means2D=torch.zeros((CROWD, 4), device="cuda", requires_grad=True),
        opacities=(0.05 + 0.15 * torch.rand((CROWD, 1), generator=g)).cuda(),
        scales=torch.full((CROWD, 3), 0.25, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(CROWD, 1).cuda(),
        colors_precomp=torch.rand((CROWD, 3), generator=g).cuda())


def test_depth_ladder_scene_spans_more_than_one_bucket():
    scene = _depth_ladder_scene()
    _color, radii, _counts, _depth, alpha = _render_crowd(scene)
    assert (radii > 0).sum() > 32, "need >32 splats in one tile for >1 bucket"
    assert alpha.max() > 0.5, "the splats must actually cover pixels"


def test_depth_position_gradients_survive_the_multi_bucket_replay():
    """The position path, checked against finite differences across buckets."""
    scene = _depth_ladder_scene(requires_grad=True)
    target = _depth_target(seed=11)

    _depth_loss(scene, target).backward()
    analytic = scene["means3D"].grad[:, 2].clone()

    worst = 0.0
    checked = 0
    for i in range(0, CROWD, 11):
        numeric = []
        for sign in (+1.0, -1.0):
            perturbed = _depth_ladder_scene()
            with torch.no_grad():
                perturbed["means3D"][i, 2] += sign * Z_EPS
            numeric.append(_depth_loss(perturbed, target).item())
        finite_difference = (numeric[0] - numeric[1]) / (2 * Z_EPS)
        scale = max(0.5, abs(analytic[i].item()), abs(finite_difference))
        worst = max(worst, abs(analytic[i].item() - finite_difference) / scale)
        checked += 1

    assert checked > 5
    assert worst < 0.15, "worst relative gradient error {:.4f}".format(worst)


def test_colour_and_depth_gradients_add_up():
    """A combined loss must give exactly the sum of the two separate gradients.

    Guards the wiring rather than the maths: if the depth channel leaked into
    the colour replay, or the colour background term were applied to the depth
    channel, the two would stop being additive.
    """
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    def grad_of(use_colour, use_depth):
        g = _one_gaussian(z=3.0)
        g["opacities"] = g["opacities"].clone().requires_grad_(True)
        out = GaussianRasterizer(_settings())(**g)
        total = 0.0
        if use_colour:
            total = total + out[0].double().sum()
        if use_depth:
            total = total + out[3].double().sum()
        total.backward()
        return g["opacities"].grad.item()

    colour_only = grad_of(True, False)
    depth_only = grad_of(False, True)
    both = grad_of(True, True)
    assert abs(both - (colour_only + depth_only)) <= 1e-3 * max(1.0, abs(both))
