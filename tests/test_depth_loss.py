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

from utils.loss_utils import (
    ALPHA_THRESHOLD,
    compute_edge_weights,
    depth_loss_fn,
    depth_supervision_mask,
    edge_aware_logl1_loss,
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
