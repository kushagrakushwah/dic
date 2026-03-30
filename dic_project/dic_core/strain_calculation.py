"""
Strain computation from tracked displacement fields.

Fix for Issue #3:
  The original code looked up neighboring points by exact integer coordinates,
  which broke silently when floating-point rounding produced slightly different
  values.  This module replaces that with a scipy KDTree so neighbors are
  found reliably within a configurable radius.

All computation is pure numpy/scipy — no GUI code here (Issue #9).
"""
import logging
import numpy as np
from scipy.spatial import cKDTree

from utils.config import STRAIN_KDTREE_RADIUS_MULTIPLIER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-frame strain computation
# ---------------------------------------------------------------------------

def compute_frame_strains(
    initial_positions: np.ndarray,
    current_positions: np.ndarray,
    grid_step: float,
    kdtree_radius_mult: float = STRAIN_KDTREE_RADIUS_MULTIPLIER,
) -> dict:
    """
    Compute 2-D Green-Lagrange strains for every point in a single frame.

    Uses central finite differences with KDTree-based neighbor lookup so
    that small floating-point deviations in the initial grid do not cause
    silent failures.

    Args:
        initial_positions  : (N, 2) float array, reference (x, y)
        current_positions  : (N, 2) float array, deformed  (x, y); may contain NaN
        grid_step          : nominal grid spacing in pixels
        kdtree_radius_mult : radius = grid_step * this for neighbor search

    Returns:
        dict with keys 'exx', 'eyy', 'exy', 'E1', 'E2', 'von_mises'
        each being a (N,) float array (NaN where not computable).
    """
    n = len(initial_positions)
    displacements = current_positions - initial_positions  # (N, 2)
    u = displacements[:, 0]  # x-displacement
    v = displacements[:, 1]  # y-displacement

    exx = np.full(n, np.nan)
    eyy = np.full(n, np.nan)
    exy = np.full(n, np.nan)
    E1 = np.full(n, np.nan)
    E2 = np.full(n, np.nan)
    vm = np.full(n, np.nan)

    radius = grid_step * kdtree_radius_mult

    # Build KDTree on initial positions (only valid/non-NaN positions)
    valid_mask = ~np.any(np.isnan(initial_positions), axis=1)
    if np.sum(valid_mask) < 3:
        return dict(exx=exx, eyy=eyy, exy=exy, E1=E1, E2=E2, von_mises=vm)

    tree = cKDTree(initial_positions[valid_mask])
    valid_indices = np.where(valid_mask)[0]

    for i in range(n):
        if not valid_mask[i]:
            continue
        if np.any(np.isnan(current_positions[i])):
            continue

        px, py = initial_positions[i]

        # --- find neighbors in ±x direction ---
        idx_px = _find_neighbor(tree, valid_indices, px + grid_step, py, radius)
        idx_mx = _find_neighbor(tree, valid_indices, px - grid_step, py, radius)
        idx_py = _find_neighbor(tree, valid_indices, px, py + grid_step, radius)
        idx_my = _find_neighbor(tree, valid_indices, px, py - grid_step, radius)

        can_du_dx = idx_px is not None and idx_mx is not None
        can_dv_dy = idx_py is not None and idx_my is not None
        can_du_dy = idx_py is not None and idx_my is not None
        can_dv_dx = idx_px is not None and idx_mx is not None

        if not (can_du_dx and can_dv_dy):
            continue

        # Check that neighbor displacements are not NaN
        neighbors_ok = all(
            not np.any(np.isnan(current_positions[j]))
            for j in [idx_px, idx_mx, idx_py, idx_my]
            if j is not None
        )
        if not neighbors_ok:
            continue

        du_dx = (u[idx_px] - u[idx_mx]) / (2.0 * grid_step)
        dv_dy = (v[idx_py] - v[idx_my]) / (2.0 * grid_step)

        du_dy = (u[idx_py] - u[idx_my]) / (2.0 * grid_step) if can_du_dy else 0.0
        dv_dx = (v[idx_px] - v[idx_mx]) / (2.0 * grid_step) if can_dv_dx else 0.0

        exx[i] = du_dx
        eyy[i] = dv_dy
        exy[i] = 0.5 * (du_dy + dv_dx)

        # Principal strains
        tensor = np.array([[exx[i], exy[i]], [exy[i], eyy[i]]])
        eigvals = np.linalg.eigvalsh(tensor)
# ... (previous tensor math) ...
        E1[i] = eigvals[1]  # larger
        E2[i] = eigvals[0]  # smaller
        vm[i] = np.sqrt(E1[i]**2 - E1[i]*E2[i] + E2[i]**2)

    # --- ADD THIS OUTLIER FILTER BEFORE RETURNING ---
    # Reject physically impossible strains (e.g., > 100% or 1.0)
    # This instantly kills runaway edge points so they don't spike the heatmap
    max_reasonable_strain = 3.0 
    
    bad_points = (np.abs(exx) > max_reasonable_strain) | \
                 (np.abs(eyy) > max_reasonable_strain) | \
                 (np.abs(exy) > max_reasonable_strain)
                 
    exx[bad_points] = np.nan
    eyy[bad_points] = np.nan
    exy[bad_points] = np.nan
    E1[bad_points] = np.nan
    E2[bad_points] = np.nan
    vm[bad_points] = np.nan
    # ------------------------------------------------

    return dict(exx=exx, eyy=eyy, exy=exy, E1=E1, E2=E2, von_mises=vm)


def _find_neighbor(tree, valid_indices, tx, ty, radius):
    """
    Return the index (into the FULL position array) of the nearest point
    within `radius` of (tx, ty), or None if none found.
    """
    dists, idxs = tree.query([tx, ty], k=1, distance_upper_bound=radius)
    if not np.isinf(dists):
        return valid_indices[idxs]
    return None


# ---------------------------------------------------------------------------
# Summary statistics across a sequence
# ---------------------------------------------------------------------------

def compute_sequence_summary(
    strain_sequence: list,
) -> dict:
    """
    Compute mean strain per frame across a list of per-frame strain dicts.

    Returns:
        dict mapping strain key → 1-D array of length n_frames (mean values).
    """
    keys = ['exx', 'eyy', 'exy', 'E1', 'E2', 'von_mises']
    summary = {k: [] for k in keys}
    for frame_strains in strain_sequence:
        for k in keys:
            summary[k].append(np.nanmean(frame_strains[k]))
    return {k: np.array(v) for k, v in summary.items()}
