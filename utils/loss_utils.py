#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

C1 = 0.01 ** 2
C2 = 0.03 ** 2

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# --- Metric depth supervision -------------------------------------------------
#
# The rasterizer renders UNNORMALISED expected depth D = sum_i(d_i * alpha_i * T_i).
# Wherever coverage is partial, D under-reports distance (a pixel at true depth
# 3.0 with alpha 0.6 renders as 1.8), so supervising those pixels systematically
# drags geometry toward the camera -- worst at depth discontinuities, and
# silently. ALPHA_THRESHOLD gates which pixels are trustworthy enough to
# supervise; dividing by alpha to "correct" D was considered and rejected as
# numerically unstable near alpha = 0.

ALPHA_THRESHOLD = 0.95


def depth_supervision_mask(sensor_mask, alpha, threshold=ALPHA_THRESHOLD):
    """AND the sensor's valid-depth mask with high-alpha rendered coverage.

    `alpha` is the rasterizer's rendered alpha (render_pkg["alpha"]) and is
    treated as non-differentiable here -- it is only used to build a boolean
    mask, never to rescale a gradient-carrying tensor.

    sensor_mask: bool tensor, shaped [1, H, W] (or [H, W]).
    alpha:       float tensor, shaped [H, W] (or matching sensor_mask).
    Returns a bool tensor broadcastable against sensor_mask's shape.
    """
    alpha_ok = alpha.detach() > threshold
    if alpha_ok.dim() == sensor_mask.dim() - 1:
        alpha_ok = alpha_ok.unsqueeze(0)
    return sensor_mask & alpha_ok


def compute_edge_weights(rgb):
    """exp(-|grad rgb|) edge weights consumed by edge_aware_logl1_loss.

    `rgb` is a camera's ground-truth image, which is STATIC for the entire
    training run -- these weights are therefore identical on every iteration
    for a given camera. Callers that iterate (e.g. the training loop) should
    compute this once per camera and reuse the result (see
    Camera.get_depth_edge_weights) rather than paying for it every call.
    """
    grad_x = torch.abs(rgb[:, :, :-1] - rgb[:, :, 1:]).mean(0, keepdim=True)
    grad_y = torch.abs(rgb[:, :-1, :] - rgb[:, 1:, :]).mean(0, keepdim=True)
    return torch.exp(-grad_x), torch.exp(-grad_y)


def edge_aware_logl1_loss(pred, gt, rgb, mask, weights=None):
    """dn-splatter's EdgeAwareLogL1.

    log() keeps a handful of grossly-wrong pixels (windows, reflections) from
    dominating. The exp(-|grad rgb|) weight down-weights the loss at image
    edges, which is where this completed depth bleeds across silhouettes.

    `mask` should already fold in any depth-quality gating (e.g.
    depth_supervision_mask) on top of plain sensor validity -- this function
    only ever reads from `pred`/`gt`, so it never mutates the tensors it is
    given (safe to call directly on render_pkg["depth"]).

    `weights`, if given, is a precomputed `(wx, wy)` pair from
    compute_edge_weights(rgb) -- since rgb is static per camera, callers may
    cache and pass these in instead of recomputing them from `rgb` on every
    call. Defaults to None, which recomputes them from `rgb` exactly as
    before, so existing callers are unaffected.
    """
    logl1 = torch.log(1.0 + torch.abs(pred - gt))

    wx, wy = weights if weights is not None else compute_edge_weights(rgb)

    mx, my = mask[:, :, :-1], mask[:, :-1, :]
    # Multiply-and-normalise instead of boolean-mask advanced indexing
    # (`tensor[bool_mask]`): the gather form has to learn its data-dependent
    # output size before it can allocate, which forces a device->host sync on
    # every call. Multiplying by the mask and summing is equivalent -- masked-
    # out positions contribute exactly 0 -- and dividing by the (clamped)
    # surviving count avoids the sync entirely. clamp(min=1) keeps a fully-
    # false mask a finite zero rather than 0/0.
    mx_f, my_f = mx.to(logl1.dtype), my.to(logl1.dtype)
    sx = (logl1[:, :, :-1] * wx * mx_f).sum()
    sy = (logl1[:, :-1, :] * wy * my_f).sum()
    n = (mx.sum() + my.sum()).clamp(min=1)
    return (sx + sy) / n


def depth_loss_fn(name):
    """Resolve --depth_loss to a callable with signature (pred, gt, rgb, mask).

    Every variant is masked -- callers are expected to pass a `mask` that
    already combines sensor validity with the alpha gate from
    depth_supervision_mask, so none of these need to know about alpha
    directly. A fully-false mask returns a finite zero rather than NaN.
    """
    def _masked(f):
        # f must return an ELEMENTWISE (unreduced) tensor shaped like pred/gt --
        # reduction happens here, over the mask, after f runs.
        def g(pred, gt, rgb, mask, weights=None):
            # Multiply-and-normalise instead of `pred[mask]`/`gt[mask]` advanced
            # indexing (and the `mask.sum() == 0` python-bool branch, which also
            # syncs): both force a device->host sync to learn a data-dependent
            # size/branch on every call. clamp(min=1) keeps a fully-false mask a
            # finite zero rather than 0/0, matching the old early-return.
            mask_f = mask.to(pred.dtype)
            n = mask.sum().clamp(min=1)
            return (f(pred, gt) * mask_f).sum() / n
        return g

    table = {
        "edgeaware_logl1": edge_aware_logl1_loss,
        "logl1": _masked(lambda p, g: torch.log(1.0 + torch.abs(p - g))),
        "l1": _masked(lambda p, g: torch.abs(p - g)),
        "huber": _masked(lambda p, g: F.smooth_l1_loss(p, g, reduction="none")),
    }
    if name not in table:
        raise ValueError("unknown --depth_loss {!r}; choose from {}".format(
            name, sorted(table)))
    return table[name]

