"""Detección de botes de la pelota a partir de su trayectoria."""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.signal import find_peaks

from ..datatypes import BallObservation, Bounce
from ..utils.geometry import smooth_series
from .court import CourtModel


class BounceDetector:
    """
    Un bote se detecta como un máximo local de la coordenada Y en imagen
    (la pelota baja hasta tocar el suelo y vuelve a subir). En coordenadas
    de imagen, Y crece hacia abajo, por lo que un bote = pico en Y.
    """

    def __init__(self, court: CourtModel, min_prominence_px: float = 6.0,
                 smooth_window: int = 5, out_margin_m: float = 0.10):
        self.court = court
        self.min_prominence = min_prominence_px
        self.smooth_window = smooth_window
        self.out_margin = out_margin_m

    def detect(self, balls: List[BallObservation]) -> List[Bounce]:
        ys = np.array([b.y if b.visible and b.y is not None else np.nan for b in balls])
        if np.isfinite(ys).sum() < 3:
            return []

        ys_s = smooth_series(ys, self.smooth_window)
        # Rellenar NaN por interpolación para find_peaks
        valid = np.isfinite(ys_s)
        idx = np.arange(len(ys_s))
        ys_filled = np.interp(idx, idx[valid], ys_s[valid])

        peaks, _ = find_peaks(ys_filled, prominence=self.min_prominence)

        bounces: List[Bounce] = []
        for p in peaks:
            b = balls[p]
            if b.court_x is None or b.court_y is None:
                continue
            inside = self.court.is_inside(b.court_x, b.court_y, self.out_margin)
            bounces.append(
                Bounce(
                    frame=b.frame,
                    court_x=b.court_x,
                    court_y=b.court_y,
                    inside=inside,
                    side=self.court.side_of(b.court_y),
                    img_x=b.x,
                    img_y=b.y,
                )
            )
        return bounces
