"""Modelo geométrico de la PISTA COMPLETA y homografía imagen -> metros.

El vídeo muestra la pista entera (los dos fondos + la red). Sistema de
coordenadas (metros): origen en la esquina del fondo LEJANO izquierdo (dobles).
  - eje X a lo ancho   (0 .. doubles_width)
  - eje Y a lo largo   (0 .. length); fondo lejano en Y=0, red en Y=length/2,
    fondo cercano en Y=length.

La pista se ve de lado (vista panorámica): la red queda VERTICAL en el centro de
la imagen, con la media pista IZQUIERDA a un lado y la DERECHA al otro. El eje
largo (fondo a fondo) va de izquierda a derecha; el ancho va de arriba (banda
superior) a abajo (banda inferior).

Orden de los 19 keypoints (media izq -> media der; en cada línea, arriba->abajo).
DEBE coincidir con el etiquetado (tools/label_court_once.py) y el dataset YOLO:

    0  fondo IZQ banda-sup dobles  1  fondo izq banda-sup indiv
    2  fondo izq banda-inf indiv   3  fondo izq banda-inf dobles
    4  saque izq banda-sup indiv   5  saque izq 'T'   6  saque izq banda-inf indiv
    7  saque DER banda-sup indiv   8  saque der 'T'   9  saque der banda-inf indiv
    10 fondo DER banda-sup dobles  11 fondo der banda-sup indiv
    12 fondo der banda-inf indiv   13 fondo der banda-inf dobles
    --- intersecciones de las 5 líneas longitudinales con la RED (Y=length/2) ---
    14 red x dobles-sup   15 red x indiv-sup   16 red x centro
    17 red x indiv-inf    18 red x dobles-inf
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from ..datatypes import CourtFrame

SERVICE_LINE_FROM_NET = 6.40   # m

# Identidades de jugador por media pista (singles): la red divide IZQUIERDA y
# DERECHA en la imagen. court_y < length/2 = media izquierda; >= = media derecha.
LEFT_PLAYER_ID = 1
RIGHT_PLAYER_ID = 2

# Nombres y orden canónico de los 19 keypoints de la PISTA COMPLETA.
# half (left/right) = lado de la red; band (top/bottom) = banda de ancho.
# 14-18: intersecciones de las 5 líneas longitudinales con la red.
COURT_KEYPOINT_NAMES = [
    "left_baseline_top_doubles", "left_baseline_top_singles",
    "left_baseline_bottom_singles", "left_baseline_bottom_doubles",
    "left_service_top_singles", "left_service_T", "left_service_bottom_singles",
    "right_service_top_singles", "right_service_T", "right_service_bottom_singles",
    "right_baseline_top_doubles", "right_baseline_top_singles",
    "right_baseline_bottom_singles", "right_baseline_bottom_doubles",
    "net_top_doubles", "net_top_singles", "net_center",
    "net_bottom_singles", "net_bottom_doubles",
]
COURT_NUM_KP = len(COURT_KEYPOINT_NAMES)   # 19
# Flip horizontal de imagen (intercambia media izq <-> media der; mantiene bandas).
# Los 5 puntos de la red (14-18) están en el centro: se mapean a sí mismos.
COURT_FLIP_IDX = [10, 11, 12, 13, 7, 8, 9, 4, 5, 6, 0, 1, 2, 3, 14, 15, 16, 17, 18]

# El vídeo es una PANORÁMICA formada por dos cámaras (media izquierda + media
# derecha) unidas por la red. NO es una proyección en perspectiva única, así que
# una sola homografía no ajusta bien (residuos > 1.5 m). Usamos DOS homografías,
# una por media pista, y proyectamos cada punto con la del lado de la imagen en
# que cae (corte en la x de la red). Índices de keypoint de cada media pista
# (incluyen los 5 de la red 14-18, que son la frontera compartida):
LEFT_HALF_IDX = [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18]
RIGHT_HALF_IDX = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
NET_IDX = [14, 15, 16, 17, 18]


def reference_keypoints(length: float, doubles_width: float,
                        singles_width: float) -> np.ndarray:
    """Keypoints de referencia (metros) de la pista completa, orden canónico."""
    L = length
    Wd = doubles_width
    tram = (doubles_width - singles_width) / 2.0   # ancho del pasillo
    xl = tram                 # línea de individuales izq
    xr = doubles_width - tram # línea de individuales der
    xc = doubles_width / 2.0  # línea central de saque
    s_far = L / 2 - SERVICE_LINE_FROM_NET   # línea de saque lejana
    s_near = L / 2 + SERVICE_LINE_FROM_NET  # línea de saque cercana
    net = L / 2                              # red
    return np.array(
        [
            [0.0, 0.0], [xl, 0.0], [xr, 0.0], [Wd, 0.0],        # 0-3 fondo lejano
            [xl, s_far], [xc, s_far], [xr, s_far],              # 4-6 saque lejano
            [xl, s_near], [xc, s_near], [xr, s_near],           # 7-9 saque cercano
            [0.0, L], [xl, L], [xr, L], [Wd, L],                # 10-13 fondo cercano
            [0.0, net], [xl, net], [xc, net], [xr, net], [Wd, net],  # 14-18 red
        ],
        dtype=np.float32,
    )


class CourtModel:
    def __init__(self, length: float, singles_width: float = 8.23,
                 doubles_width: float = 10.97):
        self.length = length
        self.singles_width = singles_width
        self.doubles_width = doubles_width
        self.width = doubles_width          # span del eje X (para viz/zonas)
        self.tramline = (doubles_width - singles_width) / 2.0
        self.reference = reference_keypoints(length, doubles_width, singles_width)
        self._last_H: Optional[np.ndarray] = None
        # Homografías por media pista (panorámica de dos cámaras) + x de corte.
        self._H_left: Optional[np.ndarray] = None
        self._H_right: Optional[np.ndarray] = None
        self._split_x: Optional[float] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _fit(img_pts: np.ndarray, ref_pts: np.ndarray) -> Optional[np.ndarray]:
        """Homografía por mínimos cuadrados con los puntos finitos (>=4)."""
        mask = np.isfinite(img_pts).all(axis=1)
        ip, rp = img_pts[mask], ref_pts[mask]
        if len(ip) < 4:
            return None
        H, _ = cv2.findHomography(ip.astype(np.float32), rp.astype(np.float32), 0)
        return H

    def estimate_homography(self, court: CourtFrame) -> CourtFrame:
        """Calcula DOS homografías (media izq / media der) con los keypoints.

        El pano une dos cámaras por la red, así que una sola homografía no vale.
        Cada punto se proyecta luego con la homografía de su lado de la imagen.
        court.homography guarda la global (compat. + chequeo de validez)."""
        if not court.valid:
            court.homography = self._last_H
            return court

        n = min(len(court.keypoints), len(self.reference))
        img_pts = court.keypoints[:n].astype(np.float32)
        ref_pts = self.reference[:n].astype(np.float32)

        idx = np.arange(n)
        l_idx = [i for i in LEFT_HALF_IDX if i < n]
        r_idx = [i for i in RIGHT_HALF_IDX if i < n]
        self._H_left = self._fit(img_pts[l_idx], ref_pts[l_idx])
        self._H_right = self._fit(img_pts[r_idx], ref_pts[r_idx])

        # x de corte = mediana de la x (imagen) de los puntos de la red.
        net_idx = [i for i in NET_IDX if i < n]
        net_xs = img_pts[net_idx][:, 0]
        net_xs = net_xs[np.isfinite(net_xs)]
        self._split_x = float(np.median(net_xs)) if len(net_xs) else None

        # Homografía global (solo para compat. y validez).
        H = self._fit(img_pts, ref_pts)
        court.homography = H if H is not None else self._last_H
        if court.homography is not None:
            self._last_H = court.homography
        return court

    # ------------------------------------------------------------------
    def to_court(self, x: float, y: float,
                 H: Optional[np.ndarray] = None) -> Optional[tuple]:
        """Proyecta un punto imagen (px) a coordenadas de pista (metros).

        Si hay homografías por media pista, elige la del lado de la imagen en que
        cae el punto (corte en la x de la red). Si no, usa la global pasada."""
        if x is None or y is None:
            return None
        if self._split_x is not None and self._H_left is not None \
                and self._H_right is not None:
            Hsel = self._H_left if x < self._split_x else self._H_right
        else:
            Hsel = H if H is not None else self._last_H
        if Hsel is None:
            return None
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, Hsel)[0, 0]
        return float(out[0]), float(out[1])

    # ------------------------------------------------------------------
    def is_inside(self, cx: float, cy: float, margin: float = 0.0,
                  singles: bool = True) -> bool:
        """¿Está el punto dentro de los límites de la pista (+ margen)?

        singles=True usa los límites de individuales (pasillos excluidos).
        """
        x_lo = (self.tramline if singles else 0.0) - margin
        x_hi = (self.width - self.tramline if singles else self.width) + margin
        return (x_lo <= cx <= x_hi) and (-margin <= cy <= self.length + margin)

    def side_of(self, cy: float) -> str:
        """Media pista respecto a la red (y=length/2). Y<mitad = izquierda,
        Y>=mitad = derecha."""
        return "left" if cy < self.length / 2 else "right"

    def player_on_side(self, cy: float) -> int:
        """ID de jugador (1=izquierda, 2=derecha) según la media pista."""
        return LEFT_PLAYER_ID if cy < self.length / 2 else RIGHT_PLAYER_ID

    @staticmethod
    def player_for_side(side: str) -> int:
        return LEFT_PLAYER_ID if side == "left" else RIGHT_PLAYER_ID
