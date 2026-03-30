"""
Central configuration for DIC Analysis Tool.
All tunable parameters and defaults live here.
"""

# --- Subset Tracking ---
DEFAULT_SUBSET_SIZE = 31          # pixels, must be odd
DEFAULT_GRID_STEP = 10            # pixels between grid points
DEFAULT_SEARCH_MULTIPLIER = 2     # search window = subset_size * multiplier
MIN_CORRELATION_THRESHOLD = 0.85   # below this, mark point as lost (NaN)

# --- Preprocessing ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (3, 3)

# --- Strain ---
STRAIN_KDTREE_RADIUS_MULTIPLIER = 1.5  # neighbor search radius = step * this

# --- Visualization ---
HEATMAP_COLORMAP_ID = 2   # cv2.COLORMAP_JET
HEATMAP_ALPHA = 0.6
VIDEO_FPS = 10
VIDEO_CODEC = "mp4v"

# --- Supported image formats ---
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# --- Logging ---
LOG_LEVEL = "INFO"
