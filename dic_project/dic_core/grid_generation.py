"""
Analysis grid generation inside a user-defined ROI polygon.
"""
import numpy as np
import cv2
import logging
from utils.roi_tools import build_roi_mask, polygon_bounding_rect

logger = logging.getLogger(__name__)


def generate_grid(
    image_shape: tuple,
    polygon_points_img: list,
    grid_step: int,
    subset_size: int,
) -> np.ndarray:
    """
    Generate a regular grid of analysis points inside the ROI polygon.

    Points are placed on a step×step grid and filtered to those:
      - Inside the polygon mask
      - Far enough from image borders to allow a full subset extraction

    Args:
        image_shape         : (height, width) of the reference image
        polygon_points_img  : list of QPoint in image coordinates
        grid_step           : spacing between grid points (pixels)
        subset_size         : subset size (pixels, must be odd)

    Returns:
        grid_points : (N, 2) float32 array of (x, y) image coordinates,
                      or empty array if no valid points found.
    """
    if len(polygon_points_img) < 3:
        raise ValueError("ROI polygon must have at least 3 vertices.")

    mask = build_roi_mask(image_shape, polygon_points_img)
    x, y, w, h = polygon_bounding_rect(polygon_points_img)
    half = subset_size // 2
    img_h, img_w = image_shape[:2]

    grid_points = []
    for gy in range(y + half, y + h - half, grid_step):
        for gx in range(x + half, x + w - half, grid_step):
            # Must be inside image with a full subset margin
            if gx - half < 0 or gy - half < 0:
                continue
            if gx + half >= img_w or gy + half >= img_h:
                continue
            if mask[gy, gx] == 255:
                grid_points.append([gx, gy])

    if not grid_points:
        logger.warning("No grid points generated. Try smaller step or larger ROI.")
        return np.empty((0, 2), dtype=np.float32)

    result = np.array(grid_points, dtype=np.float32)
    logger.info(f"Generated {len(result)} analysis points "
                f"(step={grid_step}, subset={subset_size})")
    return result
