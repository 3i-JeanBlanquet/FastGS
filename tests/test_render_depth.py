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
    # The near splat is nearly opaque, so it swallows almost all of the
    # transmittance and the blended depth stays close to its own.
    assert depth[centre, centre] / alpha[centre, centre] < 3.0


def test_colour_output_is_unaffected_by_the_widened_checkpoint_buffer():
    # A missed sampled_ar stride update shows up as colour corruption rather
    # than as a depth error, so pin the colour of an (almost) opaque white
    # splat on a black background.
    color, _radii, _counts, _depth, alpha = _render(**_one_gaussian(z=3.0))
    opaque = alpha > 0.95
    assert opaque.sum() > 0
    for channel in range(3):
        values = color[channel][opaque]
        assert torch.allclose(values, alpha[opaque], atol=1e-5)
