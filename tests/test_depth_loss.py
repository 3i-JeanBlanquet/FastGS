#
# Coverage for the metric-depth loss and its alpha-aware masking.
#
# The rasterizer renders UNNORMALISED expected depth D = sum_i(d_i*alpha_i*T_i),
# so a partially-covered pixel under-reports distance. depth_supervision_mask
# folds "alpha > 0.95" into the sensor's own valid-depth mask so that partial
# coverage never gets supervised -- the tests below exercise that gate
# directly, in addition to the four EdgeAwareLogL1 behaviours from the plan.
#
# Pure torch, no CUDA needed: everything here operates on hand-built tensors.
#

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from utils.loss_utils import (
    ALPHA_THRESHOLD,
    compute_edge_weights,
    depth_loss_fn,
    depth_supervision_mask,
    edge_aware_logl1_loss,
    normalized_depth,
)


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


# --- Ruling A: alpha masking ---------------------------------------------------

def test_alpha_mask_excludes_low_alpha_pixels():
    # One pixel is sensor-valid but only partially covered (alpha well below
    # the 0.95 gate) and carries a huge depth error. If it leaked into the
    # loss it would dominate; the mask must drop it entirely.
    gt = torch.ones(1, 4, 4) * 3.0
    pred = gt.clone()
    pred[0, 2, 2] = 0.1  # a pixel where D under-reports because coverage is partial
    rgb = torch.zeros(3, 4, 4)

    sensor_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    alpha = torch.ones(4, 4)
    alpha[2, 2] = 0.3  # well under ALPHA_THRESHOLD

    mask = depth_supervision_mask(sensor_mask, alpha)
    assert mask[0, 2, 2].item() is False

    loss_with_bad_pixel = edge_aware_logl1_loss(pred, gt, rgb, mask)
    # Reference: same mask but the bad pixel corrected to the true depth --
    # should match, proving the bad pixel contributed nothing.
    pred_fixed = pred.clone()
    pred_fixed[0, 2, 2] = gt[0, 2, 2]
    loss_reference = edge_aware_logl1_loss(pred_fixed, gt, rgb, mask)
    assert torch.allclose(loss_with_bad_pixel, loss_reference)


def test_alpha_mask_keeps_high_alpha_pixels():
    sensor_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    alpha = torch.full((4, 4), ALPHA_THRESHOLD + 0.01)
    mask = depth_supervision_mask(sensor_mask, alpha)
    assert torch.equal(mask, sensor_mask)


def test_alpha_mask_respects_sensor_mask_too():
    sensor_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    sensor_mask[0, 0, 0] = False
    alpha = torch.ones(4, 4)  # fully confident everywhere
    mask = depth_supervision_mask(sensor_mask, alpha)
    assert mask[0, 0, 0].item() is False
    assert mask.sum().item() == 15


def test_fully_masked_input_returns_finite_zero():
    gt = torch.rand(1, 8, 8) + 1.0
    pred = gt + 5.0  # would be a huge error if it counted
    rgb = torch.rand(3, 8, 8)
    m = torch.zeros(1, 8, 8, dtype=torch.bool)

    loss = edge_aware_logl1_loss(pred, gt, rgb, m)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_fully_masked_input_is_finite_zero_for_every_depth_loss_variant():
    gt = torch.rand(1, 8, 8) + 1.0
    pred = gt + 5.0
    rgb = torch.rand(3, 8, 8)
    m = torch.zeros(1, 8, 8, dtype=torch.bool)

    for name in ("edgeaware_logl1", "logl1", "l1", "huber"):
        loss = depth_loss_fn(name)(pred, gt, rgb, m)
        assert torch.isfinite(loss), name
        assert loss.item() == 0.0, name


# --- depth_loss_fn dispatch -----------------------------------------------------

def test_depth_loss_fn_dispatches_every_documented_variant():
    gt = torch.ones(1, 8, 8)
    pred = gt + 0.5
    rgb = torch.rand(3, 8, 8)
    m = torch.ones(1, 8, 8, dtype=torch.bool)

    for name in ("edgeaware_logl1", "logl1", "l1", "huber"):
        loss = depth_loss_fn(name)(pred, gt, rgb, m)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0


def test_depth_loss_fn_rejects_unknown_name():
    try:
        depth_loss_fn("not-a-real-loss")
    except ValueError:
        pass
    else:
        assert False, "expected ValueError for an unknown --depth_loss name"


# --- Cached edge weights: the optimisation must not change the result -----------
#
# edge_aware_logl1_loss recomputed exp(-|grad rgb|) from `rgb` on every call, even
# though `rgb` (a camera's ground-truth image) is static for the whole training
# run. compute_edge_weights() lets a caller (Camera.get_depth_edge_weights) do
# that work once per camera instead of once per iteration. These tests prove the
# cached path is bit-for-bit equivalent to the original always-recompute path --
# an optimisation that changes the loss value is a bug, not a speed-up.

def test_compute_edge_weights_matches_inline_formula():
    rgb = torch.rand(3, 10, 14)
    wx, wy = compute_edge_weights(rgb)

    grad_x = torch.abs(rgb[:, :, :-1] - rgb[:, :, 1:]).mean(0, keepdim=True)
    grad_y = torch.abs(rgb[:, :-1, :] - rgb[:, 1:, :]).mean(0, keepdim=True)
    assert torch.allclose(wx, torch.exp(-grad_x))
    assert torch.allclose(wy, torch.exp(-grad_y))


def test_edge_aware_logl1_loss_with_precomputed_weights_matches_fresh_computation():
    pred = torch.rand(1, 12, 12) + 1.0
    gt = torch.rand(1, 12, 12) + 1.0
    rgb = torch.rand(3, 12, 12)
    mask = torch.ones(1, 12, 12, dtype=torch.bool)

    loss_fresh = edge_aware_logl1_loss(pred, gt, rgb, mask)
    loss_cached = edge_aware_logl1_loss(pred, gt, rgb, mask, weights=compute_edge_weights(rgb))
    assert torch.allclose(loss_fresh, loss_cached)


def test_depth_loss_fn_accepts_optional_weights_kwarg_for_every_variant():
    # weights is only consumed by edgeaware_logl1; the other variants must
    # accept and ignore it so train.py can pass it unconditionally.
    gt = torch.ones(1, 8, 8)
    pred = gt + 0.5
    rgb = torch.rand(3, 8, 8)
    m = torch.ones(1, 8, 8, dtype=torch.bool)
    weights = compute_edge_weights(rgb)

    for name in ("edgeaware_logl1", "logl1", "l1", "huber"):
        loss = depth_loss_fn(name)(pred, gt, rgb, m, weights=weights)
        assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_camera_caches_depth_edge_weights_matching_fresh_computation():
    from scene.cameras import Camera

    image = torch.rand(3, 16, 16)
    cam = Camera(colmap_id=0, R=np.eye(3), T=np.zeros(3), FoVx=1.0, FoVy=1.0,
                 image=image, gt_alpha_mask=None, image_name="test", uid=0)

    wx_cached, wy_cached = cam.get_depth_edge_weights()
    wx_fresh, wy_fresh = compute_edge_weights(cam.original_image.cuda())
    assert torch.allclose(wx_cached, wx_fresh)
    assert torch.allclose(wy_cached, wy_fresh)

    # Second call must reuse the cached tensors, not recompute them.
    wx_again, wy_again = cam.get_depth_edge_weights()
    assert wx_again is wx_cached
    assert wy_again is wy_cached


# --- Sync-free masked reduction: multiply+normalise must match the old gather form
#
# `pred[mask]` / `gt[mask]` boolean-mask advanced indexing (and the
# `mask.sum() == 0` python-bool branch guarding it) forces a device->host
# synchronisation on every call, because torch has to learn the actual
# data-dependent output size/branch before it can proceed. edge_aware_logl1_loss
# and depth_loss_fn's other variants now multiply by the mask and normalise by
# its count instead, which needs no such sync. These tests pin the new sync-free
# reduction against the old gather form it replaced, on random partially-masked
# data -- an optimisation that changes the loss value is a bug, not a speed-up.

def _old_gather_edge_aware_logl1(pred, gt, rgb, mask):
    logl1 = torch.log(1.0 + torch.abs(pred - gt))
    wx, wy = compute_edge_weights(rgb)
    mx, my = mask[:, :, :-1], mask[:, :-1, :]
    lx = (logl1[:, :, :-1] * wx)[mx]
    ly = (logl1[:, :-1, :] * wy)[my]
    n = lx.numel() + ly.numel()
    if n == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return (lx.sum() + ly.sum()) / n


def _old_gather_masked(f):
    def g(pred, gt, rgb, mask):
        if mask.sum() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return f(pred[mask], gt[mask])
    return g


_OLD_GATHER_TABLE = {
    "edgeaware_logl1": _old_gather_edge_aware_logl1,
    "logl1": _old_gather_masked(lambda p, g: torch.log(1.0 + torch.abs(p - g)).mean()),
    "l1": _old_gather_masked(lambda p, g: torch.abs(p - g).mean()),
    "huber": _old_gather_masked(lambda p, g: F.smooth_l1_loss(p, g)),
}


def test_sync_free_masked_reduction_matches_old_gather_form_for_every_variant():
    torch.manual_seed(0)
    gt = torch.rand(1, 20, 24) + 1.0
    pred = gt + torch.randn(1, 20, 24) * 0.3
    rgb = torch.rand(3, 20, 24)
    mask = torch.rand(1, 20, 24) > 0.4  # a random, partial mask

    for name in ("edgeaware_logl1", "logl1", "l1", "huber"):
        old = _OLD_GATHER_TABLE[name](pred, gt, rgb, mask)
        new = depth_loss_fn(name)(pred, gt, rgb, mask)
        assert torch.allclose(old, new, atol=1e-6), name


def test_sync_free_masked_reduction_matches_old_gather_form_when_fully_true():
    torch.manual_seed(1)
    gt = torch.rand(1, 8, 10) + 1.0
    pred = gt + torch.randn(1, 8, 10) * 0.2
    rgb = torch.rand(3, 8, 10)
    mask = torch.ones(1, 8, 10, dtype=torch.bool)

    for name in ("edgeaware_logl1", "logl1", "l1", "huber"):
        old = _OLD_GATHER_TABLE[name](pred, gt, rgb, mask)
        new = depth_loss_fn(name)(pred, gt, rgb, mask)
        assert torch.allclose(old, new, atol=1e-6), name


# --------------------------------------------------------------------------
# Normalised depth D / A.
#
# CUDA-free coverage of the helper itself -- shapes, the eps guard, gradient
# reaching alpha, and non-mutation of the rasterizer's saved tensors. The
# property that makes it worth doing at all (opacity invariance) needs a real
# render and lives in tests/test_depth_normalization.py.
# --------------------------------------------------------------------------


def test_normalized_depth_divides_by_alpha():
    depth = torch.tensor([[[2.4, 1.5]]])
    alpha = torch.tensor([[0.8, 0.5]])
    got = normalized_depth(depth, alpha)
    assert torch.allclose(got, torch.tensor([[[3.0, 3.0]]]))


def test_normalized_depth_broadcasts_a_bare_hw_alpha():
    depth = torch.rand(1, 4, 5) + 1.0
    alpha = torch.rand(4, 5) * 0.5 + 0.5
    got = normalized_depth(depth, alpha)
    assert got.shape == (1, 4, 5)
    assert torch.allclose(got, depth / alpha.unsqueeze(0))


def test_normalized_depth_accepts_matching_shapes_unchanged():
    depth = torch.rand(1, 4, 5) + 1.0
    alpha = torch.rand(1, 4, 5) * 0.5 + 0.5
    assert torch.allclose(normalized_depth(depth, alpha), depth / alpha)


def test_normalized_depth_is_finite_at_zero_alpha():
    """The eps guard. These pixels are always masked out, but 0/0 -> NaN would
    poison the whole reduction through the multiply-and-sum, so it must not
    happen even there."""
    depth = torch.zeros(1, 3, 3)
    alpha = torch.zeros(3, 3)
    got = normalized_depth(depth, alpha)
    assert torch.isfinite(got).all()
    assert float(got.abs().max()) == 0.0


def test_normalized_depth_never_produces_nan_in_a_masked_reduction():
    depth = torch.rand(1, 6, 6)
    alpha = torch.zeros(6, 6)
    alpha[0, 0] = 0.99
    gt = torch.ones(1, 6, 6)
    rgb = torch.zeros(3, 6, 6)
    mask = (alpha > ALPHA_THRESHOLD).unsqueeze(0)
    pred = normalized_depth(depth, alpha)
    for name in ("edgeaware_logl1", "logl1", "l1", "huber"):
        out = depth_loss_fn(name)(pred, gt, rgb, mask)
        assert torch.isfinite(out), name


def test_normalized_depth_does_not_mutate_its_inputs():
    """Both inputs are saved for the rasterizer's backward pass."""
    depth = torch.rand(1, 4, 4) + 1.0
    alpha = torch.rand(4, 4) * 0.5 + 0.5
    d0, a0 = depth.clone(), alpha.clone()
    normalized_depth(depth, alpha)
    assert torch.equal(depth, d0) and torch.equal(alpha, a0)


def test_normalized_depth_carries_gradient_into_alpha():
    """The point of un-marking alpha non-differentiable: without a gradient
    path into alpha the division would be cosmetic."""
    depth = (torch.rand(1, 4, 4) + 1.0).requires_grad_(True)
    alpha = (torch.rand(4, 4) * 0.5 + 0.5).requires_grad_(True)
    normalized_depth(depth, alpha).sum().backward()
    assert alpha.grad is not None and alpha.grad.abs().sum() > 0
    assert depth.grad is not None and depth.grad.abs().sum() > 0


def test_depth_supervision_mask_still_blocks_gradient_through_the_mask():
    """Masking must stay detached even now that alpha is differentiable."""
    alpha = (torch.rand(4, 4) * 0.1 + 0.9).requires_grad_(True)
    sensor = torch.ones(1, 4, 4, dtype=torch.bool)
    mask = depth_supervision_mask(sensor, alpha)
    assert not mask.requires_grad


# --------------------------------------------------------------------------
# --depth_normalize.
#
# Defaults to True (the new behaviour) and must be switchable OFF so the old
# un-normalised path stays runnable and comparable. argparse's "store_true"
# cannot express that -- a True default is stuck on -- so ParamGroup registers
# True-defaulted bools with a word-parsing type instead.
# --------------------------------------------------------------------------


def _optimization_args(argv):
    from argparse import ArgumentParser

    from arguments import OptimizationParams

    parser = ArgumentParser()
    op = OptimizationParams(parser)
    return op.extract(parser.parse_args(argv))


def test_depth_normalize_defaults_to_false():
    # Measured on the reference capture: normalisation moved surface opacity
    # 0.450 -> 0.425 and floaters 6.1% -> 6.5%. Default follows the measurement.
    assert _optimization_args([]).depth_normalize is False


def test_depth_normalize_is_off_unless_asked_for():
    # Now that the default is False, ParamGroup registers it as a bare
    # store_true flag, so absence means un-normalised supervision.
    assert _optimization_args([]).depth_normalize is False
    assert _optimization_args(["--lambda_depth", "0.25"]).depth_normalize is False


def test_depth_normalize_bare_flag_turns_it_on():
    assert _optimization_args(["--depth_normalize"]).depth_normalize is True


def test_depth_normalize_does_not_swallow_the_next_option():
    args = _optimization_args(["--depth_normalize", "--lambda_depth", "0.25"])
    assert args.depth_normalize is True
    assert args.lambda_depth == 0.25


def test_false_defaulted_bools_keep_the_bare_flag_form():
    """The store_true path must be untouched for every other boolean."""
    from argparse import ArgumentParser

    from arguments import ModelParams

    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = lp.extract(parser.parse_args(["-s", "/tmp/x", "--eval", "--init_from_depth"]))
    assert args.eval is True and args.init_from_depth is True
    args = lp.extract(parser.parse_args(["-s", "/tmp/x"]))
    assert args.eval is False and args.init_from_depth is False
