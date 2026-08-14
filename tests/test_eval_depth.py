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


def test_empty_mask_is_nan_not_a_crash():
    gt = torch.ones(1, 4, 4)
    pred = gt + 5.0
    m = torch.zeros(1, 4, 4, dtype=torch.bool)
    r = depth_metrics(pred, gt, m, 4000.0)
    assert r["n_valid"] == 0
    assert r["mae_m"] != r["mae_m"]   # NaN
    assert r["rmse_m"] != r["rmse_m"]  # NaN
