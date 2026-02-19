"""Pixel-level face comparison using mask intersection and MAE/MSE metrics."""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _erode_mask(mask: np.ndarray, shrink_ratio: float = 0.05) -> np.ndarray:
    """Erode a boolean mask inward by shrink_ratio of its equivalent radius.

    Args:
        mask: (H, W) boolean array
        shrink_ratio: fraction to shrink (0.05 = 5%)

    Returns:
        (H, W) boolean array (eroded)
    """
    area = mask.sum()
    if area == 0:
        return mask
    # equivalent radius = sqrt(area / pi)
    equiv_radius = np.sqrt(area / np.pi)
    kernel_size = max(3, int(equiv_radius * shrink_ratio) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def compare_faces(
    input_img: np.ndarray,
    output_img: np.ndarray,
    input_mask: np.ndarray,
    output_mask: np.ndarray,
    mask_shrink_ratio: float = 0.05,
) -> dict:
    """Compare face regions between input and output images.

    Args:
        input_img: HWC uint8 RGB numpy array (input image)
        output_img: HWC uint8 RGB numpy array (output image)
        input_mask: (H, W) boolean mask for input face region
        output_mask: (H, W) boolean mask for output face region
        mask_shrink_ratio: shrink the intersection mask by this ratio (0.05 = 5%)

    Returns:
        dict with mae, mse, mask_iou, mask_area_ratio, pixel counts
    """
    # Resize output to input size if different
    if output_img.shape[:2] != input_img.shape[:2]:
        h, w = input_img.shape[:2]
        output_img = np.array(
            Image.fromarray(output_img).resize((w, h), Image.LANCZOS)
        )
        output_mask = np.array(
            Image.fromarray(output_mask.astype(np.uint8) * 255).resize(
                (w, h), Image.NEAREST
            )
        ) > 127

    # Compute intersection mask, then erode to remove boundary noise
    intersection_mask = input_mask & output_mask
    if mask_shrink_ratio > 0:
        intersection_mask = _erode_mask(intersection_mask, mask_shrink_ratio)
    union_mask = input_mask | output_mask

    input_mask_pixels = int(input_mask.sum())
    output_mask_pixels = int(output_mask.sum())
    intersection_pixels = int(intersection_mask.sum())
    union_pixels = int(union_mask.sum())

    # Compute IoU
    mask_iou = intersection_pixels / union_pixels if union_pixels > 0 else 0.0

    # Compute mask area ratio (intersection / input_mask)
    mask_area_ratio = (
        intersection_pixels / input_mask_pixels if input_mask_pixels > 0 else 0.0
    )

    # Extract pixels from intersection region
    if intersection_pixels == 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "mask_iou": 0.0,
            "mask_area_ratio": 0.0,
            "input_mask_pixels": input_mask_pixels,
            "output_mask_pixels": output_mask_pixels,
            "intersection_pixels": 0,
        }

    input_pixels = input_img[intersection_mask].astype(np.float64)
    output_pixels = output_img[intersection_mask].astype(np.float64)

    diff = input_pixels - output_pixels
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "mask_iou": round(mask_iou, 4),
        "mask_area_ratio": round(mask_area_ratio, 4),
        "input_mask_pixels": input_mask_pixels,
        "output_mask_pixels": output_mask_pixels,
        "intersection_pixels": intersection_pixels,
    }


def generate_diff_heatmap(
    input_img: np.ndarray,
    output_img: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Generate a difference heatmap image for the masked face region.

    Pixels with larger differences are shown in red on the original image.

    Args:
        input_img: HWC uint8 RGB numpy array
        output_img: HWC uint8 RGB numpy array (same size as input)
        mask: (H, W) boolean mask for the comparison region

    Returns:
        HWC uint8 RGB numpy array (heatmap overlay)
    """
    # Resize output to input size if different
    if output_img.shape[:2] != input_img.shape[:2]:
        h, w = input_img.shape[:2]
        output_img = np.array(
            Image.fromarray(output_img).resize((w, h), Image.LANCZOS)
        )
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (w, h), Image.NEAREST
            )
        ) > 127

    # Compute per-pixel absolute difference (grayscale)
    diff = np.abs(input_img.astype(np.float64) - output_img.astype(np.float64))
    diff_gray = np.mean(diff, axis=2)  # (H, W)

    # Normalize to 0-255
    max_diff = diff_gray.max()
    if max_diff > 0:
        diff_normalized = (diff_gray / max_diff * 255).astype(np.uint8)
    else:
        diff_normalized = np.zeros_like(diff_gray, dtype=np.uint8)

    # Apply colormap (COLORMAP_JET: blue=low, red=high)
    heatmap_bgr = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original input image, only in masked region
    result = input_img.copy()
    alpha = 0.6
    masked_area = mask[:, :, np.newaxis]
    result = np.where(
        masked_area,
        (input_img.astype(np.float64) * (1 - alpha) + heatmap_rgb.astype(np.float64) * alpha).astype(np.uint8),
        result,
    )

    return result
