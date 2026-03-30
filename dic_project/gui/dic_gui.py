"""
Main GUI for the DIC Analysis Tool.

Fixes / improvements:
  Issue #1  – CoordinateMapper provides a clean display↔image coordinate transform.
  Issue #8  – GUI code is isolated here; computation lives in dic_core / visualization.
  Issue #9  – Worker threads never touch the GUI; results are received via signals.
  Issue #10 – Parameter validation, informative status messages, and error dialogs.
"""
import os
import sys
import logging

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QGroupBox,
    QSpinBox, QProgressBar, QMessageBox, QComboBox,
    QCheckBox, QDoubleSpinBox, QGridLayout, QTabWidget,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QEvent

from gui.mpl_canvas import MplCanvas
from image_processing.image_loader import load_image_sequence
from image_processing.preprocessing import PreprocessingConfig, preprocess_sequence
from dic_core.grid_generation import generate_grid
from dic_core.subset_tracking import SubsetCorrelationWorker
from visualization.video_generator import StrainVideoWorker
from utils.roi_tools import CoordinateMapper
from utils.config import (
    DEFAULT_SUBSET_SIZE, DEFAULT_GRID_STEP,
    DEFAULT_SEARCH_MULTIPLIER, MIN_CORRELATION_THRESHOLD,
)

logger = logging.getLogger(__name__)


class DIC_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIC Analysis Tool — Refactored")
        self.setGeometry(100, 100, 1450, 920)

        # --- State ---
        self.images_gray: list = []
        self.images_gray_proc: list = []   # preprocessed
        self.images_color: list = []
        self.image_names: list = []
        self.initial_points_grid: np.ndarray = None
        self.tracked_points_df: pd.DataFrame = None
        self.strain_summary_df: pd.DataFrame = None

        # ROI
        self.roi_polygon_img: list = []    # QPoints in IMAGE coordinates
        self.is_defining_roi: bool = False
        self.current_roi_mode: str = "Polygon (Manual)"

        # Coordinate mapping (Issue #1)
        self.coord_mapper = CoordinateMapper()

        self._init_ui()
        self._init_status_bar()

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left panel — image preview + plot
        left = QVBoxLayout()
        self.image_preview_label = QLabel("Load images to begin")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setStyleSheet(
            "border: 2px solid #555; background: #2a2a2a; color: #aaa;"
        )
        self.image_preview_label.setMinimumSize(820, 600)
        self.image_preview_label.installEventFilter(self)
        left.addWidget(self.image_preview_label)

        roi_info = QHBoxLayout()
        self.roi_status_label = QLabel("ROI: Not selected")
        self.roi_dims_label = QLabel("Points: 0")
        roi_info.addWidget(self.roi_status_label)
        roi_info.addWidget(self.roi_dims_label)
        left.addLayout(roi_info)

        self.plot_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        left.addWidget(self.plot_canvas)
        root.addLayout(left, 2)

        # Right panel — tabs
        self.tabs = QTabWidget()
        self._build_setup_tab()
        self._build_analysis_tab()
        self._build_results_tab()
        root.addWidget(self.tabs, 1)

    def _build_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Output folder
        grp = QGroupBox("Output Settings")
        g = QHBoxLayout()
        self.output_folder_edit = QLineEdit(os.path.join(os.getcwd(), "DIC_Output"))
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._select_output_folder)
        g.addWidget(QLabel("Output folder:"))
        g.addWidget(self.output_folder_edit)
        g.addWidget(btn)
        grp.setLayout(g)
        layout.addWidget(grp)

        # Image folder
        grp2 = QGroupBox("Load Images")
        g2 = QGridLayout()
        self.image_folder_edit = QLineEdit()
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._select_image_folder)
        b_load = QPushButton("Load Images")
        b_load.clicked.connect(self._load_images)
        g2.addWidget(QLabel("Image folder:"), 0, 0)
        g2.addWidget(self.image_folder_edit, 0, 1)
        g2.addWidget(b_browse, 0, 2)
        g2.addWidget(b_load, 1, 0, 1, 3)
        self.load_status_label = QLabel("")
        g2.addWidget(self.load_status_label, 2, 0, 1, 3)
        grp2.setLayout(g2)
        layout.addWidget(grp2)

        # Preprocessing (Issue #5)
        grp3 = QGroupBox("Preprocessing (for colored images)")
        g3 = QGridLayout()
        self.chk_clahe = QCheckBox("CLAHE contrast enhancement")
        self.chk_clahe.setChecked(True)
        self.chk_blur = QCheckBox("Gaussian blur")
        self.chk_histeq = QCheckBox("Histogram equalisation")
        g3.addWidget(self.chk_clahe, 0, 0)
        g3.addWidget(self.chk_blur, 1, 0)
        g3.addWidget(self.chk_histeq, 2, 0)
        grp3.setLayout(g3)
        layout.addWidget(grp3)

        # ROI
        grp4 = QGroupBox("ROI Selection")
        g4 = QGridLayout()
        g4.addWidget(QLabel("Shape:"), 0, 0)
        self.roi_shape_combo = QComboBox()
        self.roi_shape_combo.addItems(["Polygon (Manual)", "Rectangle"])
        self.roi_shape_combo.currentTextChanged.connect(self._on_roi_shape_change)
        g4.addWidget(self.roi_shape_combo, 0, 1)
        self.btn_start_roi = QPushButton("Start ROI Selection")
        self.btn_start_roi.clicked.connect(self._start_roi_selection)
        self.btn_start_roi.setEnabled(False)
        g4.addWidget(self.btn_start_roi, 1, 0)
        self.btn_finish_roi = QPushButton("Finish Polygon")
        self.btn_finish_roi.clicked.connect(self._finish_roi_selection)
        self.btn_finish_roi.setEnabled(False)
        g4.addWidget(self.btn_finish_roi, 1, 1)
        grp4.setLayout(g4)
        layout.addWidget(grp4)

        # Grid parameters
        grp5 = QGroupBox("Analysis Grid")
        g5 = QGridLayout()
        g5.addWidget(QLabel("Subset size (px, odd):"), 0, 0)
        self.spin_subset = QSpinBox()
        self.spin_subset.setMinimum(9)
        self.spin_subset.setMaximum(201)
        self.spin_subset.setValue(DEFAULT_SUBSET_SIZE)
        self.spin_subset.setSingleStep(2)
        g5.addWidget(self.spin_subset, 0, 1)

        g5.addWidget(QLabel("Grid step (px):"), 1, 0)
        self.spin_step = QSpinBox()
        self.spin_step.setMinimum(2)
        self.spin_step.setMaximum(200)
        self.spin_step.setValue(DEFAULT_GRID_STEP)
        g5.addWidget(self.spin_step, 1, 1)

        self.btn_gen_grid = QPushButton("Generate Analysis Grid")
        self.btn_gen_grid.clicked.connect(self._generate_grid)
        self.btn_gen_grid.setEnabled(False)
        g5.addWidget(self.btn_gen_grid, 2, 0, 1, 2)
        grp5.setLayout(g5)
        layout.addWidget(grp5)

        layout.addStretch()
        self.tabs.addTab(tab, "1. Setup")

    def _build_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Tracking options (Issue #6)
        grp = QGroupBox("Tracking Options")
        g = QGridLayout()
        g.addWidget(QLabel("Search window multiplier:"), 0, 0)
        self.spin_search_mult = QDoubleSpinBox()
        self.spin_search_mult.setMinimum(1.0)
        self.spin_search_mult.setMaximum(10.0)
        self.spin_search_mult.setSingleStep(0.5)
        self.spin_search_mult.setValue(DEFAULT_SEARCH_MULTIPLIER)
        g.addWidget(self.spin_search_mult, 0, 1)

        g.addWidget(QLabel("Min correlation threshold:"), 1, 0)
        self.spin_corr_thresh = QDoubleSpinBox()
        self.spin_corr_thresh.setMinimum(0.0)
        self.spin_corr_thresh.setMaximum(1.0)
        self.spin_corr_thresh.setSingleStep(0.05)
        self.spin_corr_thresh.setDecimals(2)
        self.spin_corr_thresh.setValue(MIN_CORRELATION_THRESHOLD)
        g.addWidget(self.spin_corr_thresh, 1, 1)

        self.chk_reinit = QCheckBox("Reinitialize lost points from reference")
        self.chk_reinit.setChecked(True)
        g.addWidget(self.chk_reinit, 2, 0, 1, 2)
        grp.setLayout(g)
        layout.addWidget(grp)

        self.btn_start_track = QPushButton("▶  Start Subset Tracking")
        self.btn_start_track.setEnabled(False)
        self.btn_start_track.clicked.connect(self._start_tracking)
        layout.addWidget(self.btn_start_track)

        self.pb_tracking = QProgressBar()
        layout.addWidget(self.pb_tracking)

        # Visualisation
        grp2 = QGroupBox("Visualisation")
        g2 = QGridLayout()
        g2.addWidget(QLabel("Type:"), 0, 0)
        self.vis_combo = QComboBox()
        self.vis_combo.addItems(["exx", "eyy", "exy", "E1", "E2", "von_mises", "displacement"])
        self.vis_combo.currentTextChanged.connect(self._on_vis_type_change)
        g2.addWidget(self.vis_combo, 0, 1)

        self.chk_auto_scale = QCheckBox("Auto strain scale")
        self.chk_auto_scale.setChecked(True)
        self.chk_auto_scale.stateChanged.connect(self._toggle_manual_scale)
        g2.addWidget(self.chk_auto_scale, 1, 0, 1, 2)

        g2.addWidget(QLabel("Min strain:"), 2, 0)
        self.spin_min_strain = QDoubleSpinBox()
        self.spin_min_strain.setRange(-10.0, 10.0)
        self.spin_min_strain.setSingleStep(0.001)
        self.spin_min_strain.setDecimals(5)
        self.spin_min_strain.setValue(-0.01)
        self.spin_min_strain.setEnabled(False)
        g2.addWidget(self.spin_min_strain, 2, 1)

        g2.addWidget(QLabel("Max strain:"), 3, 0)
        self.spin_max_strain = QDoubleSpinBox()
        self.spin_max_strain.setRange(-10.0, 10.0)
        self.spin_max_strain.setSingleStep(0.001)
        self.spin_max_strain.setDecimals(5)
        self.spin_max_strain.setValue(0.01)
        self.spin_max_strain.setEnabled(False)
        g2.addWidget(self.spin_max_strain, 3, 1)

        grp2.setLayout(g2)
        layout.addWidget(grp2)

        self.btn_gen_vis = QPushButton("▶  Generate Visualisation Video")
        self.btn_gen_vis.setEnabled(False)
        self.btn_gen_vis.clicked.connect(self._start_vis)
        layout.addWidget(self.btn_gen_vis)

        self.pb_vis = QProgressBar()
        layout.addWidget(self.pb_vis)

        layout.addStretch()
        self.tabs.addTab(tab, "2. Analysis")
        self.tabs.setTabEnabled(1, False)

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.results_label = QLabel("Results will appear here after analysis.")
        self.results_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.results_label)

        btn_open = QPushButton("Open Output Folder")
        btn_open.clicked.connect(self._open_output_folder)
        layout.addWidget(btn_open)

        self.btn_plot_strain = QPushButton("Plot Average Strain vs. Frame")
        self.btn_plot_strain.setEnabled(False)
        self.btn_plot_strain.clicked.connect(self._plot_average_strain)
        layout.addWidget(self.btn_plot_strain)

        layout.addStretch()
        self.tabs.addTab(tab, "3. Results")
        self.tabs.setTabEnabled(2, False)

    def _init_status_bar(self):
        self.statusBar().showMessage("Ready.")

    # -------------------------------------------------------------------------
    # Slot: output folder
    # -------------------------------------------------------------------------

    def _select_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_folder_edit.setText(path)
            os.makedirs(path, exist_ok=True)
            self.statusBar().showMessage(f"Output folder: {path}")

    # -------------------------------------------------------------------------
    # Slot: load images
    # -------------------------------------------------------------------------

    def _select_image_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if path:
            self.image_folder_edit.setText(path)

    def _load_images(self):
        folder = self.image_folder_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, "Error", "Please enter an image folder path.")
            return
        try:
            color, gray, names = load_image_sequence(folder)
        except ValueError as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self.images_color = color
        self.images_gray = gray
        self.image_names = names

        # Apply preprocessing
        cfg = PreprocessingConfig(
            use_clahe=self.chk_clahe.isChecked(),
            use_blur=self.chk_blur.isChecked(),
            use_histeq=self.chk_histeq.isChecked(),
        )
        self.images_gray_proc = preprocess_sequence(gray, cfg)

        n = len(color)
        h, w = color[0].shape[:2]
        msg = f"Loaded {n} images ({w}×{h} px)"
        self.load_status_label.setText(msg)
        self.statusBar().showMessage(msg + " — Define ROI.")
        self._display_image(color[0])
        self.btn_start_roi.setEnabled(True)
        self.tabs.setTabEnabled(1, True)

    # -------------------------------------------------------------------------
    # Image display (Issue #1 — uses CoordinateMapper)
    # -------------------------------------------------------------------------

    def _display_image(self, bgr_img: np.ndarray):
        """Render bgr_img into the preview label, drawing ROI and grid."""
        if bgr_img is None:
            return
        display = bgr_img.copy()

        # Draw ROI polygon
        if self.roi_polygon_img:
            pts = np.array([[p.x(), p.y()] for p in self.roi_polygon_img], dtype=np.int32)
            closed = not self.is_defining_roi
            cv2.polylines(display, [pts], closed, (0, 220, 255), 2)
            for pt in pts:
                cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)

        # Draw grid points
        if self.initial_points_grid is not None:
            for pt in self.initial_points_grid:
                cv2.circle(display, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

        h, w = display.shape[:2]
        qimg = QImage(display.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.image_preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_preview_label.setPixmap(scaled)

        # Update coordinate mapper whenever the image/label might have changed
        lw = self.image_preview_label.width()
        lh = self.image_preview_label.height()
        self.coord_mapper.update(w, h, lw, lh)

    # -------------------------------------------------------------------------
    # ROI selection (Issue #1 — all coordinates stored in image space)
    # -------------------------------------------------------------------------

    def _on_roi_shape_change(self, text):
        self.roi_polygon_img = []
        self.is_defining_roi = False
        if self.images_color:
            self._display_image(self.images_color[0])

    def _start_roi_selection(self):
        if not self.images_color:
            QMessageBox.warning(self, "Error", "Load images first.")
            return
        self.current_roi_mode = self.roi_shape_combo.currentText()
        self.is_defining_roi = True
        self.roi_polygon_img = []
        self.initial_points_grid = None
        self.btn_gen_grid.setEnabled(False)
        self.btn_start_roi.setEnabled(False)

        if self.current_roi_mode == "Polygon (Manual)":
            self.statusBar().showMessage("Click to add polygon vertices. Click 'Finish Polygon' when done.")
            self.btn_finish_roi.setEnabled(True)
        else:
            self.statusBar().showMessage("Click the TOP-LEFT corner of the rectangle.")
            self.btn_finish_roi.setEnabled(False)
        self._display_image(self.images_color[0])

    def _finish_roi_selection(self):
        if self.current_roi_mode == "Polygon (Manual)" and len(self.roi_polygon_img) < 3:
            QMessageBox.warning(self, "Error", "Polygon needs at least 3 vertices.")
            self.roi_polygon_img = []
        self.is_defining_roi = False
        self.btn_start_roi.setEnabled(True)
        self.btn_finish_roi.setEnabled(False)
        if self.roi_polygon_img:
            self.btn_gen_grid.setEnabled(True)
            n = len(self.roi_polygon_img)
            self.roi_status_label.setText(f"ROI: {n}-vertex polygon")
            self.statusBar().showMessage("ROI defined. Click 'Generate Analysis Grid'.")
        else:
            self.statusBar().showMessage("ROI selection cancelled.")
        self._display_image(self.images_color[0])

    def eventFilter(self, source, event):
        """Handle mouse clicks on the image preview for ROI definition."""
        if source == self.image_preview_label and self.is_defining_roi:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                lx, ly = event.pos().x(), event.pos().y()

                # Refresh mapper with current label size
                if self.images_color:
                    img = self.images_color[0]
                    self.coord_mapper.update(
                        img.shape[1], img.shape[0],
                        self.image_preview_label.width(),
                        self.image_preview_label.height()
                    )

                if not self.coord_mapper.is_inside_pixmap(lx, ly):
                    return True  # Click outside displayed image area

                ix, iy = self.coord_mapper.label_to_image(lx, ly)
                self.roi_polygon_img.append(QPoint(ix, iy))
                self._display_image(self.images_color[0])

                if self.current_roi_mode == "Polygon (Manual)":
                    self.statusBar().showMessage(
                        f"{len(self.roi_polygon_img)} vertices added. Click 'Finish Polygon' when done."
                    )
                elif self.current_roi_mode == "Rectangle":
                    if len(self.roi_polygon_img) == 1:
                        self.statusBar().showMessage("Click the BOTTOM-RIGHT corner.")
                    elif len(self.roi_polygon_img) == 2:
                        p1, p2 = self.roi_polygon_img
                        self.roi_polygon_img = [
                            p1,
                            QPoint(p2.x(), p1.y()),
                            p2,
                            QPoint(p1.x(), p2.y()),
                        ]
                        self._finish_roi_selection()
                return True
        return super().eventFilter(source, event)

    # -------------------------------------------------------------------------
    # Grid generation
    # -------------------------------------------------------------------------

    def _generate_grid(self):
        if not self.images_gray or not self.roi_polygon_img:
            QMessageBox.warning(self, "Error", "Load images and define ROI first.")
            return

        subset = self.spin_subset.value()
        if subset % 2 == 0:
            subset += 1
            self.spin_subset.setValue(subset)

        try:
            grid = generate_grid(
                self.images_gray[0].shape,
                self.roi_polygon_img,
                grid_step=self.spin_step.value(),
                subset_size=subset,
            )
        except Exception as e:
            QMessageBox.critical(self, "Grid Error", str(e))
            return

        if len(grid) == 0:
            QMessageBox.warning(self, "Warning",
                "No grid points generated. Try a smaller step or larger ROI.")
            return

        self.initial_points_grid = grid
        self.roi_dims_label.setText(f"Points: {len(grid)}")
        self._display_image(self.images_color[0])
        self.statusBar().showMessage(
            f"Generated {len(grid)} analysis points. Proceed to Analysis tab."
        )
        self.tabs.setCurrentIndex(1)
        self.btn_start_track.setEnabled(True)

    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------

    def _start_tracking(self):
        if self.initial_points_grid is None or not self.images_gray_proc:
            QMessageBox.warning(self, "Error", "Generate analysis grid first.")
            return

        out = self.output_folder_edit.text()
        os.makedirs(out, exist_ok=True)

        self.btn_start_track.setEnabled(False)
        self.pb_tracking.setValue(0)
        self.statusBar().showMessage("Tracking in progress…")

        self._tracking_worker = SubsetCorrelationWorker(
            images_gray=self.images_gray_proc,
            initial_grid=self.initial_points_grid,
            subset_size=self.spin_subset.value(),
            image_names=self.image_names,
            output_folder=out,
            search_multiplier=self.spin_search_mult.value(),
            corr_threshold=self.spin_corr_thresh.value(),
            reinitialize_lost=self.chk_reinit.isChecked(),
        )
        self._tracking_worker.progress_updated.connect(self.pb_tracking.setValue)
        self._tracking_worker.tracking_finished.connect(self._on_tracking_finished)
        self._tracking_worker.error_occurred.connect(self._on_tracking_error)
        self._tracking_worker.start()

    def _on_tracking_finished(self, df: pd.DataFrame):
        self.tracked_points_df = df
        self.pb_tracking.setValue(100)
        self.btn_start_track.setEnabled(True)
        self.btn_gen_vis.setEnabled(True)
        self.statusBar().showMessage("Tracking complete. Generate a visualisation video.")
        QMessageBox.information(self, "Tracking Complete",
            "Tracking finished.\nTracked points saved to output folder.")

    def _on_tracking_error(self, msg: str):
        QMessageBox.critical(self, "Tracking Error", msg)
        self.statusBar().showMessage(f"Tracking error: {msg}")
        self.btn_start_track.setEnabled(True)
        self.pb_tracking.setValue(0)

    # -------------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------------

    def _on_vis_type_change(self, text):
        is_strain = text != 'displacement'
        self.chk_auto_scale.setEnabled(is_strain)
        manual = is_strain and not self.chk_auto_scale.isChecked()
        self.spin_min_strain.setEnabled(manual)
        self.spin_max_strain.setEnabled(manual)

    def _toggle_manual_scale(self, state):
        manual = (state != Qt.Checked)
        self.spin_min_strain.setEnabled(manual)
        self.spin_max_strain.setEnabled(manual)

    def _start_vis(self):
        if self.tracked_points_df is None:
            QMessageBox.warning(self, "Error", "Run tracking first.")
            return

        vis_type = self.vis_combo.currentText()
        vmin = self.spin_min_strain.value()
        vmax = self.spin_max_strain.value()

        if vis_type != 'displacement':
            if self.chk_auto_scale.isChecked():
                # Auto: use ±0.01 as a safe default; user can override
                vmin, vmax = -0.01, 0.01
            elif vmax <= vmin:
                QMessageBox.warning(self, "Error", "Max strain must be > Min strain.")
                return

        out = self.output_folder_edit.text()
        os.makedirs(out, exist_ok=True)

        self.btn_gen_vis.setEnabled(False)
        self.pb_vis.setValue(0)
        self.statusBar().showMessage("Generating visualisation video…")

        self._vis_worker = StrainVideoWorker(
            df_tracked=self.tracked_points_df,
            initial_positions=self.initial_points_grid,
            images_color=self.images_color,
            vis_type=vis_type,
            global_strain_min=vmin,
            global_strain_max=vmax,
            output_folder=out,
            grid_step=self.spin_step.value(),
        )
        self._vis_worker.progress_updated.connect(self.pb_vis.setValue)
        self._vis_worker.strain_finished.connect(self._on_vis_finished)
        self._vis_worker.error_occurred.connect(self._on_vis_error)
        self._vis_worker.start()

    def _on_vis_finished(self, video_path: str, summary_df: pd.DataFrame):
        self.strain_summary_df = summary_df
        self.pb_vis.setValue(100)
        self.btn_gen_vis.setEnabled(True)
        self.btn_plot_strain.setEnabled(True)
        self.tabs.setTabEnabled(2, True)
        self.results_label.setText(f"Video saved:\n{video_path}")
        self.statusBar().showMessage(f"Visualisation saved: {video_path}")
        self.tabs.setCurrentIndex(2)
        QMessageBox.information(self, "Done", f"Visualisation video saved to:\n{video_path}")

    def _on_vis_error(self, msg: str):
        QMessageBox.critical(self, "Visualisation Error", msg)
        self.statusBar().showMessage(f"Visualisation error: {msg}")
        self.btn_gen_vis.setEnabled(True)
        self.pb_vis.setValue(0)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    def _plot_average_strain(self):
        if self.strain_summary_df is None:
            return
        vis = self.vis_combo.currentText()
        if vis == 'displacement' or vis not in self.strain_summary_df.columns:
            QMessageBox.information(self, "Info",
                "Select a strain type (not 'displacement') to plot.")
            return
        y = self.strain_summary_df[vis].values
        x = np.arange(len(y))
        self.plot_canvas.plot_series(x, y,
            title=f"Average {vis} vs. Frame",
            xlabel="Frame", ylabel=f"Avg {vis}")
        self.statusBar().showMessage(f"Plotting average {vis}.")

    def _open_output_folder(self):
        path = self.output_folder_edit.text()
        if os.path.exists(path):
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        else:
            QMessageBox.warning(self, "Error", f"Folder not found: {path}")

    # -------------------------------------------------------------------------
    # Resize event — redraw after label resize to keep ROI correct
    # -------------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.images_color:
            self._display_image(self.images_color[0])
