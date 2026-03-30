"""
Preprocessing pipeline for DIC images.

Addresses Issue #5: Colored images lose contrast when naively
converted to grayscale.  This module adds CLAHE, optional blur,
and histogram equalisation to maximise speckle contrast before tracking.
"""
import cv2
import numpy as np
import logging

from utils.config import (
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, GAUSSIAN_BLUR_KERNEL
)

logger = logging.getLogger(__name__)


class PreprocessingConfig:
    """Holds user-selectable preprocessing options."""
    def __init__(
        self,
        use_clahe: bool = True,
        use_blur: bool = False,
        use_histeq: bool = False,
        clahe_clip: float = CLAHE_CLIP_LIMIT,
        clahe_tile: tuple = CLAHE_TILE_GRID_SIZE,
        blur_kernel: tuple = GAUSSIAN_BLUR_KERNEL,
    ):
        self.use_clahe = use_clahe
        self.use_blur = use_blur
        self.use_histeq = use_histeq
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.blur_kernel = blur_kernel


def preprocess_gray(gray_img: np.ndarray, cfg: PreprocessingConfig) -> np.ndarray:
    """
    Apply the preprocessing pipeline to a single grayscale image.

    Steps (all optional):
      1. CLAHE  – Contrast Limited Adaptive Histogram Equalization
      2. Gaussian blur – reduces high-frequency noise
      3. Global histogram equalization

    Returns:
        Preprocessed grayscale image (uint8).
    """
    img = gray_img.copy()

    if cfg.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=cfg.clahe_tile
        )
        img = clahe.apply(img)

    if cfg.use_blur:
        img = cv2.GaussianBlur(img, cfg.blur_kernel, 0)

    if cfg.use_histeq:
        img = cv2.equalizeHist(img)

    return img


def preprocess_sequence(
    images_gray: list,
    cfg: PreprocessingConfig,
    progress_callback=None
) -> list:
    """
    Apply preprocessing to every image in the sequence.

    Args:
        images_gray       : list of grayscale np.ndarray
        cfg               : PreprocessingConfig
        progress_callback : optional callable(int) for progress 0-100

    Returns:
        List of preprocessed grayscale images.
    """
    processed = []
    n = len(images_gray)
    for i, img in enumerate(images_gray):
        processed.append(preprocess_gray(img, cfg))
        if progress_callback:
            progress_callback(int((i + 1) / n * 100))
    logger.info(f"Preprocessed {n} images (CLAHE={cfg.use_clahe}, "
                f"blur={cfg.use_blur}, histeq={cfg.use_histeq})")
    return processed
