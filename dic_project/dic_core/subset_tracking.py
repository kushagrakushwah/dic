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
    Track one point using NCC template matching.

    Args:
        ref_image       : reference (preprocessed, grayscale)
        cur_image       : current frame (preprocessed, grayscale)
        ref_pt          : reference position [x, y]
        last_pt         : last known position [x, y] (initial guess)
        half_subset     : half of the subset size
        search_half     : half of the search window
        corr_threshold  : minimum acceptable NCC score

    Returns:
        (new_x, new_y, score) where x/y are floats (NaN on failure).
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

    # cv2.TM_CCOEFF_NORMALIZED = 5 (integer fallback for older/headless OpenCV builds)
    _TM_CCOEFF_NORM = getattr(cv2, 'TM_CCOEFF_NORMALIZED', 5)
    result = cv2.matchTemplate(search_roi, template, _TM_CCOEFF_NORM)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < corr_threshold:
        return np.nan, np.nan, float(max_val)

    new_x = sx0 + max_loc[0] + half_subset
    new_y = sy0 + max_loc[1] + half_subset
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
        reinitialize_lost: bool = True,
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
        ref_image = self.images_gray[0]
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

            cur_image = self.images_gray[frame_idx]
            current_points = np.full_like(previous_points, np.nan)
            row = {'Image': self.image_names[frame_idx]}

            for i in range(n_points):
                ref_pt = self.initial_grid[i]
                last_pt = previous_points[i]

                # If the previous position was lost, optionally reinitialize
                if np.any(np.isnan(last_pt)):
                    if self.reinitialize_lost:
                        last_pt = ref_pt
                    else:
                        row[f'Point_{i+1}_X'] = np.nan
                        row[f'Point_{i+1}_Y'] = np.nan
                        continue

                nx, ny, score = _track_single_point(
                    ref_image, cur_image, ref_pt, last_pt,
                    half_subset, search_half, self.corr_threshold
                )

                current_points[i] = [nx, ny]
                row[f'Point_{i+1}_X'] = nx
                row[f'Point_{i+1}_Y'] = ny

                if np.isnan(nx):
                    logger.debug(f"Frame {frame_idx}, point {i}: lost (score={score:.3f})")

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
