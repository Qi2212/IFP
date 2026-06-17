"""
Mask Conversion Utilities
===========================
Functions for converting between pixel masks and patch masks,
decoding bitmap annotations, and mask-to-patch-grid mapping.
"""

import io
import zlib
import base64
import numpy as np
from PIL import Image
from typing import Tuple


def decode_bitmap_data(
    bitmap_obj: dict,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Decode a base64+zlib encoded bitmap annotation into a binary mask.

    Args:
        bitmap_obj: Annotation dictionary with 'data' (base64) and 'origin' [x, y].
        image_size: (H, W) of the original image.

    Returns:
        np.ndarray: Binary mask of shape (H, W), dtype uint8.
    """
    b64 = bitmap_obj.get("data", "")
    origin = bitmap_obj.get("origin", [0, 0])

    H, W = image_size

    if not b64:
        return np.zeros((H, W), dtype=np.uint8)

    raw = base64.b64decode(b64)

    try:
        dec = zlib.decompress(raw)
        img = Image.open(io.BytesIO(dec)).convert("L")
    except Exception:
        img = Image.open(io.BytesIO(raw)).convert("L")

    arr = np.array(img)
    mask = (arr > 127).astype(np.uint8)

    canvas = np.zeros((H, W), dtype=np.uint8)
    h, w = mask.shape
    ox, oy = origin
    x0 = int(ox)
    y0 = int(oy)
    x1 = min(W, x0 + w)
    y1 = min(H, y0 + h)

    canvas[y0:y1, x0:x1] = mask[:(y1 - y0), :(x1 - x0)]

    return canvas


def pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray,
    resize: Tuple[int, int] = (512, 512),
    patch_size: int = 16,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert a pixel-level binary mask to a patch-level binary mask.

    Args:
        pixel_mask: Binary mask of shape (H_orig, W_orig), values 0 or 1.
        resize: (H, W) resize dimensions used during feature extraction.
        patch_size: Patch size of the vision transformer.
        threshold: Fraction of pixels in a patch that must be 1 for patch=1.

    Returns:
        np.ndarray: Patch mask of shape (gh, gw), dtype uint8.
    """
    H_resize, W_resize = resize

    mask_img = Image.fromarray(
        (pixel_mask * 255).astype(np.uint8)
    ).resize(
        (W_resize, H_resize),
        resample=Image.NEAREST,
    )

    mask_arr = np.array(mask_img).astype(np.float32) / 255.0

    gh = H_resize // patch_size
    gw = W_resize // patch_size

    patch_mask = np.zeros((gh, gw), dtype=np.uint8)

    for i in range(gh):
        for j in range(gw):
            y0 = i * patch_size
            x0 = j * patch_size
            block = mask_arr[y0:y0 + patch_size, x0:x0 + patch_size]
            if block.mean() >= threshold:
                patch_mask[i, j] = 1

    return patch_mask


def mask_to_patch_mask(
    mask: np.ndarray,
    h: int,
    w: int,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """
    Map a full-resolution binary SAM mask to a patch-grid binary mask.

    Args:
        mask: Binary mask of shape (orig_h, orig_w).
        h: Number of patches in the vertical direction.
        w: Number of patches in the horizontal direction.
        orig_w: Original image width.
        orig_h: Original image height.

    Returns:
        np.ndarray: Flattened boolean array of shape (h*w,).
    """
    patch_mask = np.zeros((h, w), dtype=bool)

    for py in range(h):
        y1 = int(py * orig_h / h)
        y2 = int((py + 1) * orig_h / h)
        for px in range(w):
            x1 = int(px * orig_w / w)
            x2 = int((px + 1) * orig_w / w)
            sub = mask[y1:y2, x1:x2]
            if sub.size > 0 and sub.max() > 0:
                patch_mask[py, px] = True

    return patch_mask.reshape(-1)


def compute_iou_and_dice(pred_mask, gt_mask) -> Tuple[float, float]:
    """
    Compute IoU and Dice score between two binary masks.

    Args:
        pred_mask: Predicted binary mask (PIL Image or np.ndarray).
        gt_mask: Ground-truth binary mask (PIL Image or np.ndarray).

    Returns:
        tuple: (iou, dice) as floats.
    """
    pred = np.array(pred_mask).astype(bool)
    gt = np.array(gt_mask).astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0

    return iou, dice
