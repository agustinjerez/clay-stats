"""Estructuras de datos compartidas por el pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BallObservation:
    frame: int
    x: Optional[float] = None          # px en imagen (centroide)
    y: Optional[float] = None
    court_x: Optional[float] = None    # metros en modelo de pista
    court_y: Optional[float] = None
    score: float = 0.0
    visible: bool = False
    interpolated: bool = False         # posición rellenada (no detección real)


@dataclass
class PlayerObservation:
    frame: int
    track_id: int
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 (px)
    foot_x: float                              # punto de contacto con suelo (px)
    foot_y: float
    court_x: Optional[float] = None
    court_y: Optional[float] = None
    score: float = 0.0


@dataclass
class CourtFrame:
    frame: int
    keypoints: np.ndarray                      # (N, 2) px
    homography: Optional[np.ndarray] = None    # 3x3 img -> court (metros)
    valid: bool = False


@dataclass
class Bounce:
    frame: int
    court_x: float
    court_y: float
    inside: bool                               # dentro de los límites de pista
    side: str                                  # "left" | "right"
    img_x: Optional[float] = None              # posición en imagen (px) del bote
    img_y: Optional[float] = None


@dataclass
class Shot:
    frame: int
    player_id: int
    court_x: float
    court_y: float
    ball_speed_kmh: Optional[float] = None


@dataclass
class Rally:
    start_frame: int
    end_frame: int
    shots: List[Shot] = field(default_factory=list)
    ended_by: str = "gap"                       # "gap" | "double_bounce" | "out"
    error_player: Optional[int] = None
