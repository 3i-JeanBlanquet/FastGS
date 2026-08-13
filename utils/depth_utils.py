import numpy as np
import torch
from PIL import Image


def load_depth(path, scale_mm_per_unit, resolution, depth_max_m):
    """Load a uint16 millimetre depth PNG as (depth_in_colmap_units, valid_mask).

    Masking happens in millimetres so the thresholds stay physical; conversion
    to COLMAP units comes after. Depth is planar Z -- no radial correction.

    Returns:
        depth: torch.FloatTensor [1, H, W] in COLMAP units (0 where invalid)
        mask:  torch.BoolTensor  [1, H, W], True where the pixel is valid
    """
    img = Image.open(path)
    if img.size != tuple(resolution):
        # NEAREST: interpolating across a depth discontinuity invents surfaces
        # that exist at neither depth.
        img = img.resize(resolution, Image.NEAREST)
    mm = np.asarray(img).astype(np.float32)

    valid = (mm > 0) & (mm < depth_max_m * 1000.0)
    units = mm / float(scale_mm_per_unit)
    units[~valid] = 0.0

    depth = torch.from_numpy(units).unsqueeze(0).contiguous()
    mask = torch.from_numpy(valid).unsqueeze(0).contiguous()
    return depth, mask
