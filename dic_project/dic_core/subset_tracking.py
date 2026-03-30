"""
Subset-based DIC tracking using normalised cross-correlation.

Fixes / improvements vs. original:
  - Issue #2 : Robust boundary checks — subsets near edges are safely skipped.
  - Issue #6 : Adjustable search window; low-correlation detections marked NaN;
               optional reinitialization of lost points from the reference frame.
  - Issue #9 : Pure computation — no GUI or Matplotlib calls; results are
               returned via PyQt signals only.
"""
import logging
import os

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from utils.config import (
    DEFAULT_SEARCH_MULTIPLIER,
    MIN_CORRELATION_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_extract_template(image: np.ndarray, cx: int, cy: int, half: int):
    """
    Extract a square template centered on (cx, cy) with half-size `half`.

    Returns:
        template (np.ndarray) or None if the patch would be out of bounds.
    """
    h, w = image.shape[:2]
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half + 1, cy + half + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return image[y0:y1, x0:x1]


def _track_single_point(
    ref_image: np.ndarray,
    cur_image: np.ndarray,
    ref_pt: np.ndarray,
    last_pt: np.ndarray,
    half_subset: int,
    search_half: int,
    corr_threshold: float,
) -> tuple:
    """
    Track one point using NCC template matching with Sub-pixel interpolation.
    """
    x0, y0 = int(round(ref_pt[0])), int(round(ref_pt[1]))
    template = _safe_extract_template(ref_image, x0, y0, half_subset)
    if template is None:
        return np.nan, np.nan, 0.0

    # Define search region in current image
    lx, ly = int(round(last_pt[0])), int(round(last_pt[1]))
    h, w = cur_image.shape[:2]

    sx0 = max(0, lx - search_half)
    sy0 = max(0, ly - search_half)
    sx1 = min(w, lx + search_half + template.shape[1])
    sy1 = min(h, ly + search_half + template.shape[0])

    search_roi = cur_image[sy0:sy1, sx0:sx1]

    if search_roi.shape[0] < template.shape[0] or search_roi.shape[1] < template.shape[1]:
        return np.nan, np.nan, 0.0

    _TM_CCOEFF_NORM = getattr(cv2, 'TM_CCOEFF_NORMALIZED', 5)
    result = cv2.matchTemplate(search_roi, template, _TM_CCOEFF_NORM)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < corr_threshold:
        return np.nan, np.nan, float(max_val)

    # --- SUB-PIXEL PARABOLIC FIT ---
    mx, my = max_loc
    dx, dy = 0.0, 0.0
    
    # Ensure we are not on the absolute edge of the result matrix
    if 0 < mx < result.shape[1] - 1 and 0 < my < result.shape[0] - 1:
        c = result[my, mx]
        l = result[my, mx - 1]
        r = result[my, mx + 1]
        u = result[my - 1, mx]
        d = result[my + 1, mx]
        
        # Parabolic peak approximation
        denom_x = 2 * (l - 2 * c + r)
        if denom_x != 0: 
            dx = (l - r) / denom_x
            
        denom_y = 2 * (u - 2 * c + d)
        if denom_y != 0: 
            dy = (u - d) / denom_y

    new_x = sx0 + mx + dx + half_subset
    new_y = sy0 + my + dy + half_subset
    
    return float(new_x), float(new_y), float(max_val)


# ---------------------------------------------------------------------------
# QThread worker
# ---------------------------------------------------------------------------

class SubsetCorrelationWorker(QThread):
    """
    Worker thread for subset-based DIC tracking.

    Signals:
        progress_updated(int)          – 0-100 progress
        tracking_finished(DataFrame)   – tracked positions for all frames
        error_occurred(str)            – error message
    """
    progress_updated = pyqtSignal(int)
    tracking_finished = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        images_gray: list,
        initial_grid: np.ndarray,
        subset_size: int,
        image_names: list,
        output_folder: str,
        search_multiplier: float = DEFAULT_SEARCH_MULTIPLIER,
        corr_threshold: float = MIN_CORRELATION_THRESHOLD,
        reinitialize_lost: bool = False,
    ):
        super().__init__()
        self.images_gray = images_gray
        self.initial_grid = initial_grid.copy()
        self.subset_size = subset_size
        self.image_names = image_names
        self.output_folder = output_folder
        self.search_multiplier = search_multiplier
        self.corr_threshold = corr_threshold
        self.reinitialize_lost = reinitialize_lost

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            self.error_occurred.emit(f"Subset tracking failed: {e}")

    def _run(self):
            half_subset = self.subset_size // 2
            search_half = int(self.subset_size * self.search_multiplier)
            n_points = len(self.initial_grid)
            n_frames = len(self.images_gray)

            all_frames_data = []

            # Frame 0 — initial positions
            row0 = {'Image': self.image_names[0]}
            for i, pt in enumerate(self.initial_grid):
                row0[f'Point_{i+1}_X'] = pt[0]
                row0[f'Point_{i+1}_Y'] = pt[1]
            all_frames_data.append(row0)

            previous_points = self.initial_grid.copy().astype(float)

            for frame_idx in range(1, n_frames):
                if self.isInterruptionRequested():
                    logger.info("Tracking interrupted by user.")
                    break

                # --- INCREMENTAL TRACKING UPDATE ---
                # Extract the template from the PREVIOUS frame instead of frame 0
                prev_image = self.images_gray[frame_idx - 1]
                cur_image = self.images_gray[frame_idx]
                current_points = np.full_like(previous_points, np.nan)
                row = {'Image': self.image_names[frame_idx]}

                for i in range(n_points):
                    ref_pt = self.initial_grid[i]
                    last_pt = previous_points[i]

                    if np.any(np.isnan(last_pt)):
                        if self.reinitialize_lost:
                            last_pt = ref_pt
                        else:
                            row[f'Point_{i+1}_X'] = np.nan
                            row[f'Point_{i+1}_Y'] = np.nan
                            continue

                    # Pass last_pt twice so it extracts the template around the 
                    # point's last known location from the previous frame
                    nx, ny, score = _track_single_point(
                        prev_image, cur_image, last_pt, last_pt,
                        half_subset, search_half, self.corr_threshold
                    )

                    current_points[i] = [nx, ny]
                    row[f'Point_{i+1}_X'] = nx
                    row[f'Point_{i+1}_Y'] = ny

                all_frames_data.append(row)
                previous_points = current_points.copy()

                progress = int(frame_idx / (n_frames - 1) * 100)
                self.progress_updated.emit(progress)

            df = pd.DataFrame(all_frames_data)
            os.makedirs(self.output_folder, exist_ok=True)
            csv_path = os.path.join(self.output_folder, "subset_tracked_points.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Tracking complete. Saved to {csv_path}")
            self.tracking_finished.emit(df)