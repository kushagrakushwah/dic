"""
Fast per-frame strain heatmap rendering using OpenCV.
Features a dynamic Distance Mask to prevent Convex Hull artifacts.
"""
import logging
import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy.spatial import cKDTree  # <-- NEW: Used for the distance mask

from utils.config import HEATMAP_COLORMAP_ID, HEATMAP_ALPHA

logger = logging.getLogger(__name__)


def render_strain_heatmap(
    background_bgr: np.ndarray,
    points_xy: np.ndarray,
    strain_values: np.ndarray,
    vmin: float,
    vmax: float,
    alpha: float = HEATMAP_ALPHA,
    colormap_id: int = HEATMAP_COLORMAP_ID,
) -> np.ndarray:
    """
    Render a strain heatmap overlaid on `background_bgr` using OpenCV.
    """
    h, w = background_bgr.shape[:2]
    valid = ~np.isnan(strain_values) & ~np.any(np.isnan(points_xy), axis=1)

    if np.sum(valid) < 4:
        return background_bgr.copy()

    vpts = points_xy[valid]
    vvals = strain_values[valid]

    # Interpolate onto a full pixel grid
    grid_x, grid_y = np.mgrid[0:w, 0:h]
    grid_z = _interpolate_with_fallback(vpts, vvals, grid_x, grid_y)  # shape (W, H)

    # ====================================================================
    # THE SILVER BULLET: DYNAMIC DISTANCE MASK
    # This prevents the heatmap from stretching into giant triangles over the background.
    # ====================================================================
    tree = cKDTree(vpts)
    
    # 1. Find the median spacing between your actual tracking dots
    nn_dists, _ = tree.query(vpts, k=2)
    median_spacing = np.median(nn_dists[:, 1])
    
    # 2. Set a strict boundary (e.g., don't paint anything further than 1.8x the dot spacing)
    max_render_dist = median_spacing * 3.0 
    
    # 3. Check every pixel's distance to the nearest dot
    pixel_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    pixel_dists, _ = tree.query(pixel_coords)
    pixel_dists = pixel_dists.reshape(w, h)
    
    # 4. If a pixel is in empty space, erase it (make it NaN)
    grid_z[pixel_dists > max_render_dist] = np.nan
    # ====================================================================

    # Clamp and normalise to 0-255
    grid_z_clipped = np.clip(grid_z, vmin, vmax)
    if vmax > vmin:
        scaled = (grid_z_clipped - vmin) / (vmax - vmin) * 255
        scaled = np.where(np.isnan(scaled), 0, scaled)
        norm_z = scaled.astype(np.uint8)
    else:
        norm_z = np.zeros(grid_z.shape, dtype=np.uint8)

    norm_z = norm_z.T

    # Apply colour map
    colored = cv2.applyColorMap(norm_z, colormap_id)

    # Build a valid-region mask
    mask = ~np.isnan(grid_z.T)
    colored[~mask] = 0

    # Blend
    result = background_bgr.copy()
    if np.any(mask):
        mask3 = np.stack([mask, mask, mask], axis=-1)
        blended = cv2.addWeighted(background_bgr, 1 - alpha, colored, alpha, 0)
        result = np.where(mask3, blended, background_bgr)

    return result.astype(np.uint8)


def _interpolate_with_fallback(points, values, grid_x, grid_y):
    """Try cubic → linear → nearest interpolation."""
    for method in ('cubic', 'linear', 'nearest'):
        try:
            result = griddata(points, values, (grid_x, grid_y), method=method)
            if not np.all(np.isnan(result)):
                return result
        except Exception as e:
            logger.debug(f"griddata method='{method}' failed: {e}")
    return np.full_like(grid_x, np.nan, dtype=float)


def render_displacement_arrows(
    background_bgr: np.ndarray,
    initial_positions: np.ndarray,
    current_positions: np.ndarray,
    color: tuple = (0, 255, 255),
    thickness: int = 1,
    tip_length: float = 0.3,
) -> np.ndarray:
    """Draw displacement arrow vectors."""
    frame = background_bgr.copy()
    n = len(initial_positions)
    for i in range(n):
        cp = current_positions[i]
        if np.any(np.isnan(cp)):
            continue
        start = tuple(initial_positions[i].astype(int))
        end = tuple(cp.astype(int))
        cv2.arrowedLine(frame, start, end, color, thickness, tipLength=tip_length)
    return frame