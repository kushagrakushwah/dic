"""
Image loading utilities.
"""
import os
import logging
import cv2
import numpy as np

from utils.config import SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)


def load_image_sequence(folder_path: str):
    """
    Load all supported images from a folder in sorted order.

    Returns:
        images_color  : list of BGR np.ndarray
        images_gray   : list of grayscale np.ndarray (pre-preprocessing copy)
        image_names   : list of filenames
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid image folder: {folder_path}")

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
    ])

    if not files:
        raise ValueError(f"No supported images found in: {folder_path}")

    images_color = []
    images_gray = []
    image_names = []
    failed = []

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)
        if img is None:
            logger.warning(f"Could not read: {filepath}")
            failed.append(filename)
            continue
        images_color.append(img)
        images_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        image_names.append(filename)

    if not images_color:
        raise ValueError("Failed to load any images from the specified folder.")

    if failed:
        logger.warning(f"Skipped unreadable files: {failed}")

    logger.info(f"Loaded {len(images_color)} images from '{folder_path}'")
    return images_color, images_gray, image_names
