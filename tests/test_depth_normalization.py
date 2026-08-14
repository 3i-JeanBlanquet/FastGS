#
# The property normalised depth supervision exists for: OPACITY INVARIANCE.
#
# The rasterizer renders D = sum_i(d_i * alpha_i * T_i) and A = 1 - T_final.
# Supervising D lets a misplaced splat cut its depth error two ways -- move onto
# the surface, or shrink its alpha so it contributes less to D. Fading is the
# cheaper gradient, so the optimiser fades, and the trained scene comes out
# correct in expectation but made of semi-transparent splats. That is what the
# user saw as holes in surfaces up close.
#
# Supervising D / A takes the second option away. These tests assert that
# directly rather than asserting anything about the training numbers:
#
#   * for contributors at a common depth, D / A equals that depth EXACTLY, for
#     any opacities whatsoever, while D scales with them;
#   * consequently the normalised loss has ZERO gradient w.r.t. opacity there,
#     while the un-normalised loss has a large one -- and the position gradient
#     survives in both, so the geometry signal is not what got removed;
#   * on a scene spanning many depths the invariance is approximate rather than
#     exact, and is still an order of magnitude tighter than raw D.
#
# The zero-gradient test is the sharpest check on the CUDA term: the exact
# cancellation it asserts only happens if the new dL/dA term has the RIGHT SIGN
# and the RIGHT MAGNITUDE. Flipping the sign doubles the gradient instead of
# cancelling it; dropping the term leaves it at full size.
#

import pytest
import torch

from tests.test_render_depth import TILE, _render_crowd, _settings
from utils.loss_utils import depth_loss_fn, normalized_depth

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHEET = 80          # >32 splats in the single 16x16 tile -> bucket index >= 1
SHEET_Z = 3.0


def _sheet_scene(opacity_scale=1.0, requires_grad=False):
    """80 coplanar splats at z = SHEET_Z, spread across one 16x16 tile.

    Coplanar is the point: D = sum_i(SHEET_Z * alpha_i * T_i) = SHEET_Z * A, so
    D / A is SHEET_Z exactly regardless of what the alphas are. Opacities are
    modest so that A stays well short of 1 -- a saturated scene would make the
    two formulations agree trivially.
    """
    g = torch.Generator(device="cpu").manual_seed(5)
    xy = (torch.rand((SHEET, 2), generator=g) - 0.5) * 2.4
    z = torch.full((SHEET, 1), SHEET_Z)
    opacities = (0.15 + 0.20 * torch.rand((SHEET, 1), generator=g)) * opacity_scale
    return dict(
        means3D=torch.cat([xy, z], dim=1).cuda().requires_grad_(requires_grad),
        means2D=torch.zeros((SHEET, 4), device="cuda", requires_grad=True),
        opacities=opacities.cuda().requires_grad_(requires_grad),
        scales=torch.full((SHEET, 3), 0.12, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(SHEET, 1).cuda(),
        colors_precomp=torch.rand((SHEET, 3), generator=g).cuda())


def test_sheet_scene_is_partially_covered_and_spans_buckets():
    """Without this the invariance tests below could be passing trivially."""
    _color, radii, _counts, _depth, alpha = _render_crowd(_sheet_scene())
    assert (radii > 0).sum() > 32, "need >32 splats in one tile for >1 bucket"
    assert alpha.max() > 0.3, "the splats must cover pixels"
    assert alpha.mean().item() < 0.9, \
        "alpha must not saturate, or D and D/A agree trivially"


# --------------------------------------------------------------------------
# Forward invariance.
# --------------------------------------------------------------------------

SCALES = (1.0, 0.6, 0.3)


def test_normalised_depth_is_invariant_to_a_uniform_opacity_rescale():
    """D / A is unchanged; D is not. This is the whole design in one assertion."""
    # Pixel set fixed by the FAINTEST render, so every pixel compared is
    # genuinely covered in all three.
    faint_alpha = _render_crowd(_sheet_scene(opacity_scale=min(SCALES)))[4]
    covered = faint_alpha > 0.2
    assert covered.sum() > 20, "need a decent sample of covered pixels"

    raw, normalised = [], []
    for s in SCALES:
        _c, _r, _n, depth, alpha = _render_crowd(_sheet_scene(opacity_scale=s))
        raw.append(depth[covered].mean().item())
        ratio = (depth[covered] / alpha[covered])
        # Exact, not approximate: every contributor sits at the same depth.
        assert torch.allclose(ratio, torch.full_like(ratio, SHEET_Z), atol=1e-3), \
            "D/A must recover the true depth at opacity scale {}".format(s)
        normalised.append(ratio.mean().item())

    spread = (max(normalised) - min(normalised)) / SHEET_Z
    assert spread < 1e-3, "D/A moved by {:.5f} under rescaling".format(spread)
    # And the thing it replaces really does move, so the test above has content.
    assert max(raw) / min(raw) > 1.5, \
        "raw D barely moved ({:.3f} -> {:.3f}); the scene is too saturated " \
        "for this comparison to mean anything".format(min(raw), max(raw))


def test_normalised_depth_helper_matches_the_rasterizer_identity():
    """utils.loss_utils.normalized_depth must reproduce the same quantity."""
    _c, _r, _n, depth, alpha = _render_crowd(_sheet_scene())
    covered = (alpha > 0.2).unsqueeze(0)
    got = normalized_depth(depth.unsqueeze(0), alpha)
    assert got.shape == (1, TILE, TILE)
    assert torch.allclose(got[covered], torch.full_like(got[covered], SHEET_Z), atol=1e-3)


def test_normalised_depth_is_finite_where_nothing_is_drawn():
    """The eps guard. Masked out in training, but must never be inf or NaN."""
    _c, _r, _n, depth, alpha = _render_crowd(_sheet_scene(opacity_scale=0.3))
    got = normalized_depth(depth.unsqueeze(0), alpha)
    assert (alpha < 1e-6).sum() > 0, "need some genuinely empty pixels"
    assert torch.isfinite(got).all(), "normalised depth must stay finite"


# --------------------------------------------------------------------------
# Gradient invariance -- the property that actually changes training.
# --------------------------------------------------------------------------


def _supervised_loss(scene, gt_value, normalize):
    _c, _r, _n, depth, alpha = _render_crowd(scene)
    pred = depth.unsqueeze(0)
    if normalize:
        pred = normalized_depth(pred, alpha)
    gt = torch.full_like(pred, gt_value)
    mask = (alpha.detach() > 0.2).unsqueeze(0)
    rgb = torch.zeros((3, TILE, TILE), device="cuda")
    return depth_loss_fn("logl1")(pred, gt, rgb, mask)


# The sheet sits at SHEET_Z = 3.0 and the sensor says 2.0, i.e. the splats are
# too far away. Under raw D the optimiser can cut that error simply by fading
# them (D = 3.0 * A shrinks toward 2.0 as A drops) without moving anything --
# exactly the shortcut that hollows out surfaces.
GT_NEARER = 2.0


def test_unnormalised_loss_rewards_fading_splats_out():
    """Establishes the pathology, so the next test is not vacuous."""
    scene = _sheet_scene(requires_grad=True)
    _supervised_loss(scene, GT_NEARER, normalize=False).backward()
    grad = scene["opacities"].grad.squeeze(1)
    assert grad.abs().max().item() > 1e-3, "no opacity signal to speak of"
    # Negative gradient on opacity == the optimiser is pushed to lower it.
    assert grad.mean().item() < 0.0, \
        "raw D should be pushing opacity DOWN when the splats are too far"


def test_normalised_loss_has_no_opacity_gradient_on_a_coplanar_sheet():
    """The fix, stated as an equality.

    D / A = SHEET_Z identically here, so dL/dalpha must cancel to zero: the
    depth channel's alpha term and the new alpha-output term are equal and
    opposite. Getting the new term's sign wrong doubles it; omitting it leaves
    it at full size. Either way this fails.
    """
    normalised = _sheet_scene(requires_grad=True)
    _supervised_loss(normalised, GT_NEARER, normalize=True).backward()
    got = normalised["opacities"].grad.squeeze(1)

    raw = _sheet_scene(requires_grad=True)
    _supervised_loss(raw, GT_NEARER, normalize=False).backward()
    reference = raw["opacities"].grad.squeeze(1)

    # Scaled against the un-normalised gradient rather than against an absolute
    # constant, so the assertion means "the shortcut is gone", not "the numbers
    # are small".
    relative = got.abs().max().item() / max(reference.abs().max().item(), 1e-12)
    assert relative < 0.02, \
        "normalised loss still moves opacity: {:.4g} vs raw {:.4g} " \
        "(ratio {:.4f})".format(got.abs().max().item(),
                                reference.abs().max().item(), relative)


def test_normalised_loss_keeps_the_position_gradient():
    """Invariance must remove the shortcut, not the geometry signal.

    If normalisation killed dL/dz too, the loss would supervise nothing and the
    test above would pass for entirely the wrong reason.
    """
    normalised = _sheet_scene(requires_grad=True)
    _supervised_loss(normalised, GT_NEARER, normalize=True).backward()
    got = normalised["means3D"].grad[:, 2]

    raw = _sheet_scene(requires_grad=True)
    _supervised_loss(raw, GT_NEARER, normalize=False).backward()
    reference = raw["means3D"].grad[:, 2]

    assert got.abs().max().item() > 1e-4, "dL/dz must survive normalisation"
    # D / A is SHEET_Z on every supervised pixel and the sensor says GT_NEARER,
    # so dL/dz is unambiguously positive: descent pulls the sheet toward the
    # camera, which is the correction we actually want.
    assert got.sum().item() > 0.0
    assert got.abs().max().item() > 0.2 * reference.abs().max().item(), \
        "the position signal must stay comparable, not be scaled away"


# --------------------------------------------------------------------------
# The approximate case: contributors at DIFFERENT depths.
#
# The exact invariance above relies on the contributors sharing a depth. In
# general D / A is a convex combination of the contributors' depths whose
# weights shift slightly with opacity, so it moves a little. It is documented
# here rather than glossed over -- the claim being defended is that it moves far
# less than raw D does, not that it is constant.
# --------------------------------------------------------------------------

SPREAD_Z_LO, SPREAD_Z_HI = 2.5, 3.5


def _spread_scene(opacity_scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(5)
    xy = (torch.rand((SHEET, 2), generator=g) - 0.5) * 2.4
    z = SPREAD_Z_LO + (SPREAD_Z_HI - SPREAD_Z_LO) * torch.rand((SHEET, 1), generator=g)
    opacities = (0.15 + 0.20 * torch.rand((SHEET, 1), generator=g)) * opacity_scale
    return dict(
        means3D=torch.cat([xy, z], dim=1).cuda(),
        means2D=torch.zeros((SHEET, 4), device="cuda", requires_grad=True),
        opacities=opacities.cuda(),
        scales=torch.full((SHEET, 3), 0.12, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(SHEET, 1).cuda(),
        colors_precomp=torch.rand((SHEET, 3), generator=g).cuda())


def test_normalised_depth_is_far_more_stable_than_raw_depth_across_depths():
    faint_alpha = _render_crowd(_spread_scene(opacity_scale=min(SCALES)))[4]
    covered = faint_alpha > 0.2
    assert covered.sum() > 20

    raw, normalised = [], []
    for s in SCALES:
        _c, _r, _n, depth, alpha = _render_crowd(_spread_scene(opacity_scale=s))
        raw.append(depth[covered])
        normalised.append(depth[covered] / alpha[covered])
        # Still a proper depth: a convex combination of the contributors'.
        assert normalised[-1].min().item() > SPREAD_Z_LO - 0.2
        assert normalised[-1].max().item() < SPREAD_Z_HI + 0.2

    def drift(series):
        base = series[0]
        return max(((s - base).abs() / base).median().item() for s in series[1:])

    assert drift(raw) > 0.25, "raw D must move a lot, or there is nothing to fix"
    assert drift(normalised) < 0.05, \
        "D/A drifted {:.4f} across opacity scales".format(drift(normalised))
    assert drift(normalised) < 0.1 * drift(raw)
