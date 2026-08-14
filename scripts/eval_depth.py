"""Depth error on held-out (test) views, in metres.

This is the metric that actually decides whether metric depth supervision
helped: PSNR is precisely the number that cannot see a scene full of
floaters or coarse geometry error, since it only looks at colour.

`depth_metrics` is pure torch and CPU-safe -- see tests/test_eval_depth.py,
which import it directly and run without CUDA. The `__main__` block below
additionally needs a trained model and the CUDA rasterizer, so those imports
are deferred until `__main__` runs, keeping this module importable on
machines without a CUDA build (same reasoning as the sys.path fix in
scripts/estimate_depth_scale.py).
"""

import torch


def depth_metrics(pred, gt, mask, scale_mm_per_unit):
    """Depth error in metres, computed only over `mask`. Inputs are in COLMAP units.

    pred, gt, mask must be broadcastable to the same shape; mask is bool.
    Returns {"mae_m", "rmse_m", "n_valid"}. If mask has no True entries the
    error fields are NaN (nothing to average) rather than raising, so a
    camera with zero surviving pixels doesn't crash a batch evaluation --
    callers are expected to check n_valid before trusting mae_m/rmse_m.
    """
    to_m = float(scale_mm_per_unit) / 1000.0
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return {"mae_m": float("nan"), "rmse_m": float("nan"), "n_valid": 0}
    err = (pred - gt)[mask] * to_m
    return {
        "mae_m": err.abs().mean().item(),
        "rmse_m": err.pow(2).mean().sqrt().item(),
        "n_valid": n_valid,
    }


if __name__ == "__main__":
    import os
    import sys

    # Running as `python scripts/eval_depth.py` only puts scripts/ on
    # sys.path, not the repo root -- `scene`, `arguments`, etc. would not
    # otherwise be importable (identical fix to estimate_depth_scale.py).
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from argparse import ArgumentParser

    from arguments import ModelParams, PipelineParams, get_combined_args
    from fused_ssim import fused_ssim as fast_ssim
    from gaussian_renderer import GaussianModel, render_fastgs
    from scene import Scene
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    from utils.loss_utils import depth_supervision_mask

    parser = ArgumentParser(description="Depth error on held-out (test) views, in metres")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Evaluating " + args.model_path)

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # Ground-truth sensor depth lives in the dataset (it comes from the
    # capture's depth PNGs + depth_scale.json), not in the model. A baseline
    # trained with --depths="" never loaded sensor_depth onto its cameras,
    # but the depth files are still sitting on disk -- so force loading here
    # regardless of how THIS model was trained. Without this, a baseline
    # couldn't be evaluated on the same footing as a depth-supervised model,
    # and the whole point of this script is exactly that comparison.
    dataset.depths = "eval"

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        test_cameras = scene.getTestCameras()
        if len(test_cameras) == 0:
            raise RuntimeError(
                "No test cameras were loaded -- this model must have been trained "
                "(or must be reloaded) with --eval for held-out depth evaluation "
                "to mean anything.")

        n_gaussians = gaussians.get_xyz.shape[0]

        psnrs = []
        ssims = []
        maes = []
        rmses = []
        mask_fractions = []
        n_valid_total = 0
        n_no_depth = 0

        for view in test_cameras:
            render_pkg = render_fastgs(view, gaussians, pipe, background, args.mult)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(view.original_image.cuda(), 0.0, 1.0)

            psnrs.append(psnr(image, gt_image).mean().item())
            ssims.append(fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item())

            if view.sensor_depth is None:
                # No ground-truth depth for this view at all (missing
                # depth_scale.json, missing depth PNG, or flagged as an
                # outlier upstream) -- excluded from every depth statistic,
                # counted here so the report says so instead of silently
                # averaging over fewer views than it claims.
                n_no_depth += 1
                continue

            pred_depth = render_pkg["depth"].unsqueeze(0)
            alpha = render_pkg["alpha"]
            sensor_mask = view.depth_mask.to(alpha.device)
            gt_depth = view.sensor_depth.to(pred_depth.device).float()
            # Same alpha-gated mask training supervises with (see
            # utils/loss_utils.depth_supervision_mask) -- evaluating over a
            # different pixel set than training used would make the number
            # meaningless as a check on what training actually optimised.
            mask = depth_supervision_mask(sensor_mask, alpha)

            # Coverage fraction is reported for every view with ground-truth
            # depth, including a view whose mask happens to be all-False --
            # dropping those would hide exactly the "no floaters, but also no
            # coverage" failure mode this metric exists to catch.
            mask_fractions.append(mask.float().mean().item())

            m = depth_metrics(pred_depth, gt_depth, mask, scene.depth_scale)
            if m["n_valid"] > 0:
                maes.append(m["mae_m"])
                rmses.append(m["rmse_m"])
                n_valid_total += m["n_valid"]

        mean_psnr = sum(psnrs) / len(psnrs) if psnrs else float("nan")
        mean_ssim = sum(ssims) / len(ssims) if ssims else float("nan")

        print("Model         : {}".format(args.model_path))
        print("Gaussians     : {}".format(n_gaussians))
        print("Test views    : {} total, {} without ground-truth depth".format(
            len(test_cameras), n_no_depth))
        print("PSNR          : {:.3f}".format(mean_psnr))
        print("SSIM          : {:.4f}".format(mean_ssim))

        if mask_fractions:
            mean_frac = sum(mask_fractions) / len(mask_fractions)
            print("Mask fraction : {:.1%}  (sensor-valid & alpha>0.95, mean over {} views "
                  "with ground-truth depth)".format(mean_frac, len(mask_fractions)))
        else:
            print("Mask fraction : n/a -- no view had ground-truth depth")

        if maes:
            mean_mae = sum(maes) / len(maes)
            mean_rmse = sum(rmses) / len(rmses)
            print("Depth MAE (m) : {:.4f}  (mean over {} views)".format(mean_mae, len(maes)))
            print("Depth RMSE (m): {:.4f}  (mean over {} views)".format(mean_rmse, len(rmses)))
            print("Valid pixels  : {} total across those {} views. Metrics above are computed "
                  "ONLY over this pixel set -- a comparison against a run with a different valid "
                  "pixel count/mask fraction is not apples-to-apples; say so rather than reading "
                  "the numbers as directly comparable.".format(n_valid_total, len(maes)))
        else:
            print("Depth MAE (m) : n/a -- no view produced any surviving (sensor & alpha>0.95) "
                  "pixels; either no depth ground truth was available or the mask was empty "
                  "everywhere.")
            print("Depth RMSE (m): n/a")
