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

import math

import numpy as np
import pytest
import torch

from tests.test_render_depth import CROWD, FOV, TILE, ZFAR, ZNEAR, _crowded_scene, _one_gaussian, _render_crowd, _settings
from utils.graphics_utils import getProjectionMatrix, getWorld2View2

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
# Path 1 under a camera that is actually rotated.
#
# `z_view = V[2]*x + V[6]*y + V[10]*z + V[14]`, so the depth gradient reaches
# position through the THIRD ROW of the view matrix, which in the flattened
# column-major layout is (V[2], V[6], V[10]). Reading the third column,
# (V[8], V[9], V[10]), is the natural transposition mistake -- and every other
# test in this suite builds the camera as getWorld2View2(eye(3), 0), where the
# view matrix is the identity, both readings equal (0, 0, 1) and only V[10] = 1
# is exercised. Such a test cannot tell the two apart. This one can.
# --------------------------------------------------------------------------


def _two_axis_rotation():
    """A rotation whose third row and third column differ substantially."""
    ax, ay = 0.6, 0.4
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(ax), -math.sin(ax)],
                   [0.0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0.0, math.sin(ay)],
                   [0.0, 1.0, 0.0],
                   [-math.sin(ay), 0.0, math.cos(ay)]])
    return rx.dot(ry)


def _rotated_settings(R, W=64, H=64):
    from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings

    world_view_transform = torch.tensor(
        getWorld2View2(R, np.zeros(3), np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(ZNEAR, ZFAR, FOV, FOV).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
        projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    tan = math.tan(FOV * 0.5)
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


def test_rotated_camera_exercises_more_than_v10():
    """Without this the test below would be testing nothing new."""
    R = _two_axis_rotation()
    v = getWorld2View2(R, np.zeros(3), np.array([0.0, 0.0, 0.0]), 1.0).T.reshape(-1)
    correct = (v[2], v[6], v[10])
    transposed = (v[8], v[9], v[10])
    assert abs(correct[0]) > 0.3 and abs(correct[1]) > 0.3, \
        "V[2] and V[6] must be well clear of zero, else the indexing is untested"
    assert max(abs(a - b) for a, b in zip(correct, transposed)) > 0.5, \
        "the third row and third column must differ, else transposing is a no-op"


def test_position_gradient_matches_finite_difference_under_a_rotated_camera():
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    R = _two_axis_rotation()
    settings = _rotated_settings(R)
    # The camera sits at the world origin looking down its own +z, so the world
    # point at view-space depth 3 is 3 * (third column of R).
    centre = torch.tensor(3.0 * R[:, 2], dtype=torch.float32, device="cuda").reshape(1, 3)

    def rendered(delta, requires_grad):
        g = _one_gaussian(z=3.0)
        g["means3D"] = (centre + delta).clone().requires_grad_(requires_grad)
        out = GaussianRasterizer(settings)(**g)
        return out[3].sum(), g["means3D"]

    zero = torch.zeros((1, 3), device="cuda")
    total, means = rendered(zero, True)
    assert total.item() > 0.0, "the splat must be in front of the rotated camera"
    total.backward()
    analytic = means.grad[0].clone()

    eps = 0.05
    for axis in range(3):
        numeric = []
        for sign in (+1.0, -1.0):
            delta = torch.zeros((1, 3), device="cuda")
            delta[0, axis] = sign * eps
            with torch.no_grad():
                numeric.append(rendered(delta, False)[0].item())
        finite_difference = (numeric[0] - numeric[1]) / (2 * eps)
        # Score each component against the size of the whole gradient vector: a
        # transposed read redistributes the gradient between x and y rather than
        # rescaling any one component, and this catches that without letting a
        # small component's own noise blow up the ratio.
        scale = max(analytic.norm().item(), abs(finite_difference))
        error = abs(analytic[axis].item() - finite_difference) / scale
        assert error < 0.15, "axis {} analytic {:.4f} numeric {:.4f} rel {:.4f}".format(
            axis, analytic[axis].item(), finite_difference, error)


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


def _noise_floor(analytic):
    """Comparison floor, scaled to the gradients this scene actually produces.

    A splat whose own gradient is a thousandth of the scene's largest is in the
    regime where the `alpha < 1/255` cutoff and the opacity-dependent tile
    culling make the loss genuinely discontinuous, and a central difference
    there measures a step rather than a derivative. Measured on splat 132 of
    _crowded_scene, whose analytic gradient is -0.00067: the central difference
    reads -0.0107 at eps=5e-3, +0.0007 at 1e-3 and exactly 0.0 at 2e-4 and
    4e-5, i.e. it scales like step/(2*eps) instead of converging. A
    well-conditioned splat in the same scene (132 -> 22, analytic -2.8109)
    reads -2.8101 / -2.8121 / -2.8240 / -2.8246 across those same four epsilons.

    So the floor is 2% of the largest gradient present rather than a constant
    picked to fit one scene. It keeps its teeth: corrupting the replay puts the
    error at the scale of the gradients themselves, which is 50x this.
    """
    return 0.02 * analytic.abs().max().item()


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
    floor = _noise_floor(analytic)

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
        scale = max(floor, abs(analytic[i].item()), abs(finite_difference), 1e-9)
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
    floor = _noise_floor(analytic)

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
        scale = max(floor, abs(analytic[i].item()), abs(finite_difference), 1e-9)
        worst = max(worst, abs(analytic[i].item() - finite_difference) / scale)
        checked += 1

    assert checked > 5
    assert worst < 0.15, "worst relative gradient error {:.4f}".format(worst)


# --------------------------------------------------------------------------
# Resolutions the tile grid does not divide.
#
# BLOCK_X == BLOCK_Y == 16, so at 70x70 the grid of 5x5 tiles overhangs the
# image by 10 pixels on each far edge. The backward stages 32 pixels at a time
# out of `pixel_depths` -- which is the forward's `out_depth`, a bare {H, W}
# tensor with no trailing slack -- and an overhanging pixel indexes past its
# end. Every other test in this suite and in test_render_depth.py renders at
# 64x64 or 16x16, where the grid lands exactly on the image, which is precisely
# why nothing caught it.
#
# The overhanging values are never consumed (the consume path requires
# valid_pixel), so this test cannot fail on the arithmetic; it exists to give
# compute-sanitizer a workload that touches the clipped tiles, and to keep a
# non-aligned resolution permanently exercised.
# --------------------------------------------------------------------------

# 16 divides neither 70 nor 100, and does divide 48, so these cover the
# overhang on each axis independently as well as on both at once.
RAGGED = [(70, 70), (100, 48), (48, 100)]


@pytest.mark.parametrize("W,H", RAGGED)
def test_depth_backward_at_a_resolution_the_tile_grid_does_not_divide(W, H):
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    g = _one_gaussian(z=3.0, scale=5.0)
    g["means3D"] = g["means3D"].clone().requires_grad_(True)
    g["opacities"] = g["opacities"].clone().requires_grad_(True)

    out = GaussianRasterizer(_settings(W=W, H=H))(**g)
    depth, alpha = out[3], out[4]
    assert depth.shape == (H, W)
    # The far corner sits in a tile the grid only partly covers. Without this
    # the clipped tiles might never be rasterised and the test would be inert.
    assert alpha[-1, -1] > 0.5, "the splat must reach the clipped edge tiles"

    depth.sum().backward()
    assert g["means3D"].grad[0, 2].abs() > 1e-6
    assert g["opacities"].grad.abs().sum() > 1e-6


# --------------------------------------------------------------------------
# Path 3: dL/dA -> alpha -> opacity, scale and rotation.
#
# The rendered alpha A = 1 - T_final used to be marked non-differentiable,
# which was defensible while it only ever fed a boolean mask. The normalised
# depth loss D / A supervises THROUGH it, and that is the entire mechanism by
# which normalisation works: it is what makes the loss invariant to a uniform
# rescaling of the alphas, so fading a splat out stops being a way to reduce
# depth error.
#
# T_final = prod_j (1 - alpha_j), so dT_final/dalpha_i = -T_final/(1 - alpha_i)
# and dA/dalpha_i = +T_final/(1 - alpha_i). Note the sign: it is the NEGATIVE
# of the background term the colour channel already carries, which rides
# dT_final/dalpha_i directly. A sign error here is invisible to any test that
# only checks the gradient is non-zero, hence the finite differences below.
# --------------------------------------------------------------------------


def test_alpha_output_carries_gradient_at_all():
    """The regression guard for ctx.mark_non_differentiable(alpha).

    While alpha was marked non-differentiable, .backward() on it raised
    outright -- so this fails loudly rather than silently reading zeros.
    """
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    g = _one_gaussian(z=3.0, opacity=0.5)
    g["opacities"] = g["opacities"].clone().requires_grad_(True)
    out = GaussianRasterizer(_settings())(**g)
    assert out[4].requires_grad, "the alpha output must be part of the graph"
    out[4].sum().backward()
    assert g["opacities"].grad is not None
    assert g["opacities"].grad.abs().sum() > 1e-6


def test_alpha_gradient_flows_to_scale():
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    g = _one_gaussian(z=3.0, opacity=0.5)
    g["scales"] = g["scales"].clone().requires_grad_(True)
    out = GaussianRasterizer(_settings())(**g)
    out[4].sum().backward()
    assert g["scales"].grad is not None
    # A bigger splat covers more pixels, so sum(A) grows with scale. The only
    # route from A to scale is via alpha.
    assert g["scales"].grad.abs().sum() > 1e-6


def test_alpha_gradient_sign_is_positive_in_opacity():
    """A more opaque splat covers more, so d(sum A)/d(opacity) > 0.

    Flipping the sign of the new term -- the single most likely way to get it
    wrong, since the adjacent background term is the same expression negated --
    leaves the magnitude untouched and only this catches it.
    """
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    g = _one_gaussian(z=3.0, opacity=0.5)
    g["opacities"] = g["opacities"].clone().requires_grad_(True)
    out = GaussianRasterizer(_settings())(**g)
    out[4].sum().backward()
    assert g["opacities"].grad.item() > 0.0, \
        "raising opacity must raise accumulated alpha"


# --------------------------------------------------------------------------
# Multi-bucket finite differences for the alpha path.
#
# _crowded_scene and _depth_ladder_scene are both useless here: they stack 150
# splats onto the same few pixels, so T_final collapses to ~0 and dA/dalpha,
# which is T_final/(1 - alpha), is numerically zero for every splat. That would
# be a test of nothing. The scene below keeps >32 splats in ONE tile (so the
# bucket index reaches >= 1 and the sampled_ar stride arithmetic is live) while
# spreading them widely enough that coverage stays partial.
# --------------------------------------------------------------------------

COVER = 80          # ceil(80/32) = 3 buckets in the single 16x16 tile
COVER_Z_STEP = 0.02  # depth ladder: perturbing opacity must not reorder splats

# An order of magnitude below the 5e-3 the depth tests above use, and measured
# rather than guessed. The splats here are faint (alpha ~ 0.15-0.35), so plenty
# of their pixels sit near the kernel's `alpha < 1/255` cutoff and near the
# opacity-dependent compact-box tile culling, where the loss genuinely steps.
# Splat 49 shows it: at eps=5e-3 the central difference reads -0.993 against an
# analytic -0.506, and at 1e-3 and 2e-4 it reads -0.50598 and -0.50597. The
# large step straddles a discontinuity and measures the step; the smaller ones
# measure the derivative, and every sampled splat then agrees with the analytic
# gradient to better than 2e-4 absolute.
COVER_EPS = 1e-3


def _coverage_scene(requires_grad=False):
    """80 modestly-sized splats spread across one 16x16 tile. Deterministic."""
    g = torch.Generator(device="cpu").manual_seed(5)
    xy = (torch.rand((COVER, 2), generator=g) - 0.5) * 2.4
    z = 2.0 + COVER_Z_STEP * torch.arange(COVER, dtype=torch.float32).unsqueeze(1)
    return dict(
        means3D=torch.cat([xy, z], dim=1).cuda(),
        means2D=torch.zeros((COVER, 4), device="cuda", requires_grad=True),
        opacities=(0.15 + 0.20 * torch.rand((COVER, 1), generator=g)).cuda()
                  .requires_grad_(requires_grad),
        scales=torch.full((COVER, 3), 0.12, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(COVER, 1).cuda(),
        colors_precomp=torch.rand((COVER, 3), generator=g).cuda())


def test_coverage_scene_spans_buckets_without_saturating_alpha():
    """Without both properties the finite-difference test below is inert."""
    scene = _coverage_scene()
    _color, radii, _counts, _depth, alpha = _render_crowd(scene)
    assert (radii > 0).sum() > 32, "need >32 splats in one tile for >1 bucket"
    assert alpha.max() > 0.3, "the splats must actually cover pixels"
    # T_final = 1 - A must stay well clear of 0, or dA/dalpha vanishes and the
    # finite difference measures noise instead of a derivative.
    assert alpha.mean().item() < 0.9, \
        "alpha saturates -- dA/dalpha would be ~0 everywhere"


def _alpha_loss(scene, target):
    # float64 reduction: a float32 sum over the image loses enough precision
    # that a central difference suffers catastrophic cancellation.
    alpha = _render_crowd(scene)[4]
    return ((alpha.double() - target.double()) ** 2).sum()


def test_alpha_opacity_gradients_survive_the_multi_bucket_replay():
    scene = _coverage_scene(requires_grad=True)
    g = torch.Generator(device="cpu").manual_seed(13)
    target = torch.rand((TILE, TILE), generator=g).cuda()

    _alpha_loss(scene, target).backward()
    analytic = scene["opacities"].grad.clone().squeeze(1)
    floor = _noise_floor(analytic)

    worst = 0.0
    checked = 0
    for i in range(0, COVER, 7):
        numeric = []
        for sign in (+1.0, -1.0):
            perturbed = _coverage_scene()
            with torch.no_grad():
                perturbed["opacities"][i, 0] += sign * COVER_EPS
            numeric.append(_alpha_loss(perturbed, target).item())
        finite_difference = (numeric[0] - numeric[1]) / (2 * COVER_EPS)
        scale = max(floor, abs(analytic[i].item()), abs(finite_difference), 1e-9)
        worst = max(worst, abs(analytic[i].item() - finite_difference) / scale)
        checked += 1

    assert checked > 5
    # Deleting or negating the new term drives this to ~1.0 or ~2.0.
    assert worst < 0.15, "worst relative gradient error {:.4f}".format(worst)


def test_colour_depth_and_alpha_gradients_add_up():
    """Wiring guard: the three outputs must be additive in dL/dopacity.

    If the new alpha term leaked into the colour or depth replay -- crosstalk
    in the warp shuffle pipeline, or in the shared staging -- this breaks even
    though each output's own gradient still looks plausible.
    """
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    def grad_of(*outputs):
        g = _one_gaussian(z=3.0, opacity=0.5)
        g["opacities"] = g["opacities"].clone().requires_grad_(True)
        out = GaussianRasterizer(_settings())(**g)
        total = 0.0
        for i in outputs:
            total = total + out[i].double().sum()
        total.backward()
        return g["opacities"].grad.item()

    parts = grad_of(0) + grad_of(3) + grad_of(4)
    both = grad_of(0, 3, 4)
    assert abs(both - parts) <= 1e-3 * max(1.0, abs(both))


def test_alpha_gradient_is_zero_when_alpha_is_unused():
    """The depth path must be unchanged by alpha becoming differentiable.

    autograd materialises a zero grad for the unused alpha output, so a
    depth-only loss must produce exactly the gradient it produced before.
    """
    from diff_gaussian_rasterization_fastgs import GaussianRasterizer

    def grad_of(use_alpha):
        g = _one_gaussian(z=3.0, opacity=0.5)
        g["opacities"] = g["opacities"].clone().requires_grad_(True)
        out = GaussianRasterizer(_settings())(**g)
        total = out[3].double().sum()
        if use_alpha:
            total = total + 0.0 * out[4].double().sum()
        total.backward()
        return g["opacities"].grad.item()

    assert abs(grad_of(False) - grad_of(True)) <= 1e-9 * max(1.0, abs(grad_of(False)))


def test_colour_and_depth_gradients_add_up():
    """A combined loss must give exactly the sum of the two separate gradients.

    Guards the wiring rather than the maths: if the depth channel leaked into
    the colour replay -- crosstalk in ar[], in the shfl_up pipeline or in the
    shared staging -- the two would stop being additive.

    It does NOT cover the missing background term for the depth channel, which
    is a separate correctness property: every test here sets bg=zeros(3), so
    bg_dot_dpixel is identically 0 and applying it to the depth channel would
    change nothing. That would need a non-zero background to test.
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
