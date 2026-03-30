"""
Fast per-frame strain heatmap rendering using OpenCV.

Fix for Issue #4:
  The original code created a new Matplotlib figure for *every* frame,
  which is extremely slow.  This module replaces that with pure OpenCV
  operations (normalize → applyColorMap → addWeighted) which are orders
  of magnitude faster.

Fix for Issue #7:
  scipy.griddata is called with a fallback hierarchy:
    cubic → linear → nearest
  to avoid crashes on sparse or nearly-collinear point clouds.
"""
import logging
import numpy as np
import cv2
from scipy.interpolate import griddata

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

    Args:
        background_bgr : (H, W, 3) uint8 BGR background image
        points_xy      : (N, 2) float array of (x, y) point positions
        strain_values  : (N,)  float array of strain values (may contain NaN)
        vmin, vmax     : colour scale limits
        alpha          : overlay transparency (0=transparent, 1=opaque)
        colormap_id    : cv2.COLORMAP_* constant (int)

    Returns:
        (H, W, 3) uint8 BGR composite image.
    """
    h, w = background_bgr.shape[:2]
    valid = ~np.isnan(strain_values) & ~np.any(np.isnan(points_xy), axis=1)

    if np.sum(valid) < 4:
        return background_bgr.copy()

    vpts = points_xy[valid]
    vvals = strain_values[valid]

    # Interpolate onto a full pixel grid (Issue #7: cubic → linear → nearest)
    grid_x, grid_y = np.mgrid[0:w, 0:h]
    grid_z = _interpolate_with_fallback(vpts, vvals, grid_x, grid_y)  # shape (W, H)

    # Clamp and normalise to 0-255 (use nanmin to avoid NaN cast warnings)
    grid_z_clipped = np.clip(grid_z, vmin, vmax)
    if vmax > vmin:
        scaled = (grid_z_clipped - vmin) / (vmax - vmin) * 255
        # NaN pixels → 0 before casting to uint8
        scaled = np.where(np.isnan(scaled), 0, scaled)
        norm_z = scaled.astype(np.uint8)
    else:
        norm_z = np.zeros(grid_z.shape, dtype=np.uint8)

    # grid_z is (W, H) → transpose to (H, W)
    norm_z = norm_z.T

    # Apply colour map
    colored = cv2.applyColorMap(norm_z, colormap_id)

    # Build a valid-region mask (to avoid colouring extrapolated background)
    mask = ~np.isnan(grid_z.T)
    colored[~mask] = 0

    # Blend
    result = background_bgr.copy()
    if np.any(mask):
        # Only blend where we have data
        mask3 = np.stack([mask, mask, mask], axis=-1)
        blended = cv2.addWeighted(background_bgr, 1 - alpha, colored, alpha, 0)
        result = np.where(mask3, blended, background_bgr)

    return result.astype(np.uint8)


def _interpolate_with_fallback(points, values, grid_x, grid_y):
    """
    Try cubic → linear → nearest interpolation, returning the first that
    succeeds without all-NaN output.
    """
    for method in ('cubic', 'linear', 'nearest'):
        try:
            result = griddata(points, values, (grid_x, grid_y), method=method)
            if not np.all(np.isnan(result)):
                return result
        except Exception as e:
            logger.debug(f"griddata method='{method}' failed: {e}")
    # Last resort: return NaN grid
    return np.full_like(grid_x, np.nan, dtype=float)


def render_displacement_arrows(
    background_bgr: np.ndarray,
    initial_positions: np.ndarray,
    current_positions: np.ndarray,
    color: tuple = (0, 255, 255),
    thickness: int = 1,
    tip_length: float = 0.3,
) -> np.ndarray:
    """
    Draw displacement arrow vectors on a copy of `background_bgr`.
    """
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
