import matplotlib
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
import random
import torch.nn as nn


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def resize_max_res(
    img: Image.Image, max_edge_resolution: int, resample_method=Resampling.BILINEAR
) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `Image.Image`: Resized image.
    """
    original_width, original_height = img.size
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = img.resize((new_width, new_height), resample=resample_method)
    return resized_img


def get_pil_resample_method(method_str: str) -> Resampling:
    resample_method_dic = {
        "bilinear": Resampling.BILINEAR,
        "bicubic": Resampling.BICUBIC,
        "nearest": Resampling.NEAREST,
    }
    resample_method = resample_method_dic.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method

def pyramid_noise_like(x, discount=0.9):
  b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
  u = nn.Upsample(size=(w, h), mode='bilinear')
  noise = torch.randn_like(x)
  for i in range(10):
    r = random.random()*2+2 # Rather than always going 2x, 
    w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
    noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
    if w==1 or h==1: break # Lowest resolution is 1x1
  return noise/noise.std() # Scaled back to roughly unit variance

def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
