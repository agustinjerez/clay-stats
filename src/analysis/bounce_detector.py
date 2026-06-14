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
        # Trabajar SOLO con detecciones reales (incluye huecos cortos ya
        # rellenados por el refinado). NO se interpola sobre huecos largos, así
        # no se crean picos falsos por la línea recta de la interpolación.
        real = [b for b in balls if b.visible and b.y is not None]
        if len(real) < 3:
            return []

        ys = np.array([b.y for b in real], dtype=float)
        ys_s = smooth_series(ys, self.smooth_window)
        ys_s = np.where(np.isfinite(ys_s), ys_s, ys)

        # Y crece hacia abajo: un bote = la pelota baja al máximo y vuelve a subir
        peaks, _ = find_peaks(ys_s, prominence=self.min_prominence)

        bounces: List[Bounce] = []
        for k in peaks:
            b = real[k]
            if b.court_x is None or b.court_y is None:
                continue
            inside = self.court.is_inside(b.court_x, b.court_y, self.out_margin)
            bounces.append(
                Bounce(
                    frame=b.frame, court_x=b.court_x, court_y=b.court_y,
                    inside=inside, side=self.court.side_of(b.court_y),
                    img_x=b.x, img_y=b.y,
                )
            )
        return bounces
