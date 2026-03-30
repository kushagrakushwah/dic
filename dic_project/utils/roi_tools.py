"""
ROI utilities: coordinate transformations between QLabel display space
and original image pixel space. This fixes Issue #1 (ROI instability).
"""
import numpy as np
import cv2
from PyQt5.QtCore import QPoint, QSize


class CoordinateMapper:
    """
    Maps (x, y) coordinates between QLabel display space and original
    image space, accounting for KeepAspectRatio scaling and letterboxing.

    Call update() whenever the label or image size changes.
    """

    def __init__(self):
        self._img_w = 1
        self._img_h = 1
        self._label_w = 1
        self._label_h = 1
        self._x_offset = 0.0
        self._y_offset = 0.0
        self._scale_x = 1.0
        self._scale_y = 1.0

    def update(self, img_w: int, img_h: int, label_w: int, label_h: int):
        """Recompute mapping parameters."""
        self._img_w = img_w
        self._img_h = img_h
        self._label_w = label_w
        self._label_h = label_h

        # Compute the displayed pixmap size (KeepAspectRatio)
        img_aspect = img_w / img_h
        label_aspect = label_w / label_h
        if img_aspect > label_aspect:
            pix_w = label_w
            pix_h = int(label_w / img_aspect)
        else:
            pix_h = label_h
            pix_w = int(label_h * img_aspect)

        self._x_offset = (label_w - pix_w) / 2.0
        self._y_offset = (label_h - pix_h) / 2.0
        self._scale_x = img_w / pix_w if pix_w > 0 else 1.0
        self._scale_y = img_h / pix_h if pix_h > 0 else 1.0
        self._pix_w = pix_w
        self._pix_h = pix_h

    def label_to_image(self, lx: float, ly: float):
        """Convert label-space (lx, ly) to image-space (ix, iy)."""
        ix = (lx - self._x_offset) * self._scale_x
        iy = (ly - self._y_offset) * self._scale_y
        return int(np.clip(ix, 0, self._img_w - 1)), int(np.clip(iy, 0, self._img_h - 1))

    def image_to_label(self, ix: float, iy: float):
        """Convert image-space (ix, iy) to label-space (lx, ly)."""
        lx = ix / self._scale_x + self._x_offset
        ly = iy / self._scale_y + self._y_offset
        return lx, ly

    def is_inside_pixmap(self, lx: float, ly: float) -> bool:
        """Return True if the label-space point falls within the displayed pixmap."""
        return (self._x_offset <= lx < self._x_offset + self._pix_w and
                self._y_offset <= ly < self._y_offset + self._pix_h)


def build_roi_mask(image_shape: tuple, polygon_points_img: list) -> np.ndarray:
    """
    Build a binary mask (uint8) from a list of (QPoint or array) polygon
    vertices in **image** coordinates.

    Args:
        image_shape: (height, width) of the reference image.
        polygon_points_img: list of QPoint objects in image coordinates.

    Returns:
        mask: np.ndarray uint8, same HxW as image_shape.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if len(polygon_points_img) < 3:
        return mask
    pts = np.array([[p.x(), p.y()] for p in polygon_points_img], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def polygon_bounding_rect(polygon_points_img: list):
    """
    Return (x, y, w, h) bounding rect of polygon in image coordinates.
    """
    pts = np.array([[p.x(), p.y()] for p in polygon_points_img], dtype=np.int32)
    return cv2.boundingRect(pts)
