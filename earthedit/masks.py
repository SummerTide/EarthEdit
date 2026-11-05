"""Change masks: dilation, training blobs, latent grid, composite alpha."""
import math, random
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch.nn.functional as F


def dilate(mask, r):
    mask = mask.astype(bool)
    if r <= 0 or not mask.any():
        return mask
    return ndimage.distance_transform_edt(~mask) <= r


def random_blobs(h=512, w=512, rng=random):
    im = Image.new('L', (w, h), 0)
    dr = ImageDraw.Draw(im)
    for _ in range(rng.randint(1, 4)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        a, b = rng.uniform(20, 120), rng.uniform(20, 120)
        ang = math.radians(rng.uniform(0, 180))
        if rng.random() < 0.5:
            pts = [(cx + a * math.cos(u) * math.cos(ang) - b * math.sin(u) * math.sin(ang),
                    cy + a * math.cos(u) * math.sin(ang) + b * math.sin(u) * math.cos(ang)) for u in np.linspace(0, 2 * math.pi, 48)]
        else:
            pts = [(cx + sx * a * math.cos(ang) - sy * b * math.sin(ang), cy + sx * a * math.sin(ang) + sy * b * math.cos(ang))
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        dr.polygon(pts, fill=1)
    return np.asarray(im) > 0


def latent_mask(pix_mask_t):
    """(B,1,512,512) float {0,1} -> (B,1,64,64): a cell is 1 if any of its pixels is masked."""
    return F.max_pool2d(pix_mask_t, 8)


def cell_mask(lat_mask):
    return F.interpolate(lat_mask, scale_factor=8, mode='nearest')


def alpha_mask(c, dil, feather):
    """Pixel alpha of the final composite: max(blur(dilate(C)), C)."""
    d = dilate(c, dil).astype(np.float32)
    if feather > 0:
        d = ndimage.gaussian_filter(d, feather, truncate=3.0)
    return np.maximum(d, c.astype(np.float32))
