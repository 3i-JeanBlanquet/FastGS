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

import torch

from utils.loss_utils import (
    ALPHA_THRESHOLD,
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
