from __future__ import annotations

import numpy as np


def au_mass_ratio(cam: np.ndarray, region_mask: np.ndarray) -> float:
    """Fraction of CAM energy falling inside the region mask. Higher is better."""
    total = float(cam.sum())
    if total <= 0:
        return 0.0
    return float((cam * region_mask).sum() / total)


def bg_leakage(cam: np.ndarray, face_mask: np.ndarray) -> float:
    """Fraction of CAM energy outside the face. Lower is better."""
    total = float(cam.sum())
    if total <= 0:
        return 0.0
    return float((cam * (~face_mask)).sum() / total)


def pointing_game(cam: np.ndarray, region_mask: np.ndarray) -> bool:
    """Does the CAM peak fall inside the region mask?"""
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return bool(region_mask[y, x])
