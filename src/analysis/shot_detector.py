"""Detección de golpes (impactos de raqueta) y atribución por jugador.

Un GOLPE es una reversión de la dirección horizontal de la pelota: en la vista
de lado, la pelota viaja a lo largo de la pista (eje X de la imagen) entre los
dos jugadores; cuando un jugador la golpea, la X cambia de sentido. Por tanto:

    golpes = extremos locales (máximos y mínimos) de la X de la pelota en imagen.

Se trabaja sobre la X de IMAGEN (no en metros), así la detección NO depende de
la homografía de pista (que puede ser imperfecta con perspectiva). Atribución:
  - mínimo de X  -> jugador de la IZQUIERDA
  - máximo de X  -> jugador de la DERECHA
y el lado se mapea al track_id de SAM3 de ese lado.

Distinto de un BOTE, que es una reversión VERTICAL (eje Y de imagen): al botar,
la X de la pelota sigue avanzando, así que ambas señales no se confunden.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks

from ..datatypes import BallObservation, PlayerObservation, Shot
from ..utils.geometry import euclidean, smooth_series
from ..utils.logging_utils import get_logger
from .court import CourtModel, LEFT_PLAYER_ID, RIGHT_PLAYER_ID

logger = get_logger(__name__)


class ShotDetector:
    def __init__(self, court: CourtModel, fps: float,
                 min_frames_between_shots: int = 10,
                 smooth_window: int = 7,
                 max_gap_frames: int = 12,
                 prominence_frac: float = 0.12,
                 min_prominence_px: float = 25.0,
                 # compat (parámetros antiguos, ignorados)
                 min_prominence_m: float = None,
                 max_dist_to_player_m: float = None,
                 direction_change_deg: float = None):
        self.court = court
        self.fps = fps
        self.refractory = min_frames_between_shots
        self.smooth_window = smooth_window
        self.max_gap = max_gap_frames
        self.prominence_frac = prominence_frac
        self.min_prominence_px = min_prominence_px

    # ------------------------------------------------------------------
    def detect(
        self,
        balls: List[BallObservation],
        players_by_frame: Dict[int, List[PlayerObservation]],
        player_sides: Optional[Dict[str, int]] = None,
    ) -> List[Shot]:
        # Trabajamos SOLO sobre las detecciones REALES de la pelota (no
        # interpolamos), así la reversión se detecta a partir del último punto
        # antes de que la pelota desaparezca junto al jugador y el primero al
        # reaparecer: el hueco del golpe se "salta" de forma natural.
        real = [b for b in balls if b.visible and b.x is not None]
        if len(real) < 5:
            logger.warning("ShotDetector: muy pocas detecciones de pelota (%d)", len(real))
            return []

        xr = np.array([b.x for b in real], dtype=float)
        frames = np.array([b.frame for b in real])
        xs = smooth_series(xr, self.smooth_window)
        xs = np.where(np.isfinite(xs), xs, xr)

        # Prominencia de la reversión: fracción del recorrido horizontal + suelo px.
        span = float(np.nanmax(xs) - np.nanmin(xs))
        prom = max(self.min_prominence_px, self.prominence_frac * span)

        # find_peaks sobre los puntos reales: un máximo de X = la pelota iba a la
        # derecha y se devuelve (golpe del jugador derecho); un mínimo = izquierdo.
        maxima, _ = find_peaks(xs, prominence=prom)   # -> derecha
        minima, _ = find_peaks(-xs, prominence=prom)  # -> izquierda

        if player_sides is None:
            player_sides = self.player_sides(players_by_frame, self.court)

        events = sorted([(int(k), "right") for k in maxima] +
                        [(int(k), "left") for k in minima])

        shots: List[Shot] = []
        last_frame = -10_000
        for k, side in events:
            b = real[k]
            # Refractario en FRAMES (los puntos reales no son consecutivos)
            if b.frame - last_frame < self.refractory:
                continue
            player_id = player_sides.get(side, self.court.player_for_side(side))
            speed = self._speed_kmh(real, k)
            shots.append(Shot(
                frame=b.frame, player_id=player_id,
                court_x=b.court_x if b.court_x is not None else float("nan"),
                court_y=b.court_y if b.court_y is not None else float("nan"),
                ball_speed_kmh=speed,
            ))
            last_frame = b.frame
            logger.debug("Golpe @frame %d | jugador %s (%s) | x_img=%.0f | v=%s km/h",
                         b.frame, player_id, side, xs[k], speed)

        per = {pid: sum(1 for s in shots if s.player_id == pid)
               for pid in set(s.player_id for s in shots)}
        logger.info("ShotDetector: %d golpes (izq/der) desde %d reversiones X | reparto %s",
                    len(shots), len(maxima) + len(minima), per)
        return shots

    # ------------------------------------------------------------------
    @staticmethod
    def player_sides(players_by_frame, court: CourtModel = None) -> Dict[str, int]:
        """Mapea 'left'/'right' -> track_id de SAM3 según la X mediana (imagen)
        de cada jugador. Robusto: no depende de la homografía. Respaldo a IDs
        lógicos 1 (left) / 2 (right)."""
        xs: Dict[int, list] = {}
        for frame in players_by_frame.values():
            for pl in frame:
                if pl.foot_x is not None:
                    xs.setdefault(pl.track_id, []).append(pl.foot_x)
        medians = {pid: float(np.median(v)) for pid, v in xs.items() if v}
        if not medians:
            return {"left": LEFT_PLAYER_ID, "right": RIGHT_PLAYER_ID}
        order = sorted(medians, key=medians.get)     # menor foot_x = izquierda
        left_id = order[0]
        right_id = order[-1]
        if left_id == right_id:
            right_id = RIGHT_PLAYER_ID if left_id != RIGHT_PLAYER_ID else LEFT_PLAYER_ID
        return {"left": left_id, "right": right_id}

    def _speed_kmh(self, balls: List[BallObservation], p: int) -> Optional[float]:
        """Velocidad de la pelota tras el golpe (m -> km/h), si hay coords métricas."""
        if p + 3 >= len(balls):
            return None
        a, b = balls[p + 1], balls[p + 3]
        if None in (a.court_x, a.court_y, b.court_x, b.court_y):
            return None
        dist = euclidean((a.court_x, a.court_y), (b.court_x, b.court_y))
        dt = (b.frame - a.frame) / self.fps
        if dt <= 0:
            return None
        return round(dist / dt * 3.6, 1)
