"""Utilidades geométricas compartidas."""
from __future__ import annotations

import numpy as np


def smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Media móvil que ignora NaN (huecos sin detección)."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window) / window
    mask = ~np.isnan(values)
    filled = np.where(mask, values, 0.0)
    smoothed = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(mask.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smoothed / counts
    out[counts == 0] = np.nan
    return out


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Ángulo (grados) entre dos vectores 2D."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p2, float)))
