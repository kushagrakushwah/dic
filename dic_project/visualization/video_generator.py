"""
Video generation worker.

Issues addressed:
  #4  – All per-frame rendering is done with OpenCV; Matplotlib is NOT used here.
  #9  – This is a pure-computation QThread.  The GUI is updated only via signals.
"""
import os
import logging

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from dic_core.strain_calculation import compute_frame_strains, compute_sequence_summary
from visualization.heatmap import render_strain_heatmap, render_displacement_arrows
from utils.config import VIDEO_FPS, VIDEO_CODEC, HEATMAP_COLORMAP_ID, HEATMAP_ALPHA

logger = logging.getLogger(__name__)


class StrainVideoWorker(QThread):
    """
    Worker thread that:
      1. Computes per-frame strains from tracked displacement data.
      2. Renders an overlay video using fast OpenCV heatmapping.
      3. Saves a CSV of average strain per frame.
      4. Emits the finished video path and the summary DataFrame.

    Signals:
        progress_updated(int)              – 0-100
        strain_finished(str, DataFrame)    – video path, summary DF
        error_occurred(str)
    """
    progress_updated = pyqtSignal(int)
    strain_finished = pyqtSignal(str, pd.DataFrame)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        df_tracked: pd.DataFrame,
        initial_positions: np.ndarray,
        images_color: list,
        vis_type: str,
        global_strain_min: float,
        global_strain_max: float,
        output_folder: str,
        grid_step: int,
    ):
        super().__init__()
        self.df_tracked = df_tracked
        self.initial_positions = initial_positions.copy()
        self.images_color = images_color
        self.vis_type = vis_type
        self.global_strain_min = global_strain_min
        self.global_strain_max = global_strain_max
        self.output_folder = output_folder
        self.grid_step = grid_step
        self.num_points = len(initial_positions)

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            self.error_occurred.emit(f"Visualization failed: {e}")

    def _run(self):
        os.makedirs(self.output_folder, exist_ok=True)
        frame_h, frame_w = self.images_color[0].shape[:2]
        video_path = os.path.join(
            self.output_folder, f"{self.vis_type}_visualization.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (frame_w, frame_h))

        strain_sequence = []
        summary_rows = []
        n_frames = len(self.df_tracked)

        for frame_idx in range(n_frames):
            if self.isInterruptionRequested():
                break

            row = self.df_tracked.iloc[frame_idx]
            cur_pos = np.array([
                [row.get(f'Point_{i+1}_X', np.nan),
                 row.get(f'Point_{i+1}_Y', np.nan)]
                for i in range(self.num_points)
            ])

            # --- Strain computation ---
            frame_strains = compute_frame_strains(
                self.initial_positions, cur_pos, self.grid_step
            )
            strain_sequence.append(frame_strains)

            avg_row = {'Image': row['Image']}
            for k, arr in frame_strains.items():
                avg_row[k] = float(np.nanmean(arr))
            summary_rows.append(avg_row)

            # --- Rendering ---
            bg = self.images_color[frame_idx].copy()

            if self.vis_type == 'displacement':
                frame_out = render_displacement_arrows(
                    bg, self.initial_positions, cur_pos
                )
            else:
                strain_vals = frame_strains.get(self.vis_type)
                if strain_vals is None:
                    frame_out = bg
                else:
                    # If auto-scale is effectively 0 range, skip overlay
                    if self.global_strain_max <= self.global_strain_min:
                        frame_out = bg
                    else:
                        frame_out = render_strain_heatmap(
                            bg, cur_pos, strain_vals,
                            self.global_strain_min, self.global_strain_max,
                            alpha=HEATMAP_ALPHA,
                            colormap_id=HEATMAP_COLORMAP_ID,
                        )

            writer.write(frame_out)

            if n_frames > 1:
                self.progress_updated.emit(int(frame_idx / (n_frames - 1) * 100))

        writer.release()

        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(self.output_folder, "strain_results.csv")
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Video saved to {video_path}")
        self.strain_finished.emit(video_path, summary_df)
