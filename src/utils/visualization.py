"""Render de vídeo anotado con pista, jugadores y pelota."""
from __future__ import annotations

import os
from bisect import bisect_right
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..datatypes import BallObservation, Bounce, PlayerObservation
from .logging_utils import get_logger
from .video import VideoReader

logger = get_logger(__name__)

# Aristas de la PISTA COMPLETA (19 keypoints, ver court.py). Las líneas
# longitudinales se parten en la RED (puntos 14-18) en dos tramos, para que
# se dibujen bien aunque cada media pista esté en una perspectiva distinta.
COURT_EDGES = [
    (0, 3), (10, 13),                       # fondos lejano y cercano (transversal)
    (4, 6), (7, 9),                         # líneas de saque (transversal)
    (14, 15), (15, 16), (16, 17), (17, 18),  # red (5 puntos)
    (0, 14), (14, 10),                      # dobles banda sup (en 2 tramos)
    (3, 18), (18, 13),                      # dobles banda inf
    (1, 15), (15, 11),                      # individuales banda sup
    (2, 17), (17, 12),                      # individuales banda inf
    (5, 16), (16, 8),                       # central de saque (a través de la red)
]

# Colores BGR
C_COURT = (0, 255, 0)
C_COURT_KP = (0, 255, 255)
C_PLAYER = (255, 128, 0)
C_BALL = (0, 0, 255)
C_BOUNCE_IN = (255, 255, 255)   # bote dentro
C_BOUNCE_OUT = (0, 0, 255)      # bote fuera


def _draw_x(frame, c, color, r=8, t=2):
    cv2.line(frame, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), color, t, cv2.LINE_AA)
    cv2.line(frame, (c[0] - r, c[1] + r), (c[0] + r, c[1] - r), color, t, cv2.LINE_AA)


def draw_bounce_markers(frame, bounces, frame_idx, hold):
    """Dibuja los botes en su posición de imagen. Todos los pasados quedan como
    punto pequeño persistente; el reciente (<=hold) como una X grande.
    Blanco=dentro, rojo=fuera."""
    for b in bounces:
        if b.img_x is None or b.frame > frame_idx:
            continue
        c = (int(b.img_x), int(b.img_y))
        col = C_BOUNCE_IN if b.inside else C_BOUNCE_OUT
        if frame_idx - b.frame <= hold:        # reciente -> X grande
            _draw_x(frame, c, col)
            cv2.circle(frame, c, 11, col, 1, cv2.LINE_AA)
        else:                                  # pasado -> punto persistente
            cv2.circle(frame, c, 4, col, -1, cv2.LINE_AA)
    return frame


def draw_shot_markers(frame, shots, ball_by_frame, frame_idx, hold):
    """Marca cada golpe reciente (<=hold): aro en la pelota en el frame del golpe
    + rótulo 'GOLPE J{n}'. Como los golpes no se pintaban, ahora se ven."""
    recent = [s for s in shots if 0 <= frame_idx - s.frame <= hold]
    for s in recent:
        b = ball_by_frame.get(s.frame)
        if b is not None and b.x is not None:
            c = (int(b.x), int(b.y))
            label = f"GOLPE J{s.player_id}"
            if s.ball_speed_kmh:
                label += f" {s.ball_speed_kmh:.0f} km/h"
            cv2.circle(frame, c, 16, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (c[0] + 18, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return frame


def draw_court(frame: np.ndarray, kps: Optional[np.ndarray]) -> np.ndarray:
    if kps is None:
        return frame
    for a, b in COURT_EDGES:
        pa, pb = kps[a], kps[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     C_COURT, 2, cv2.LINE_AA)
    for (x, y) in kps:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(frame, (int(x), int(y)), 4, C_COURT_KP, -1, cv2.LINE_AA)
    return frame


def draw_players(frame: np.ndarray, players: List[PlayerObservation]) -> np.ndarray:
    for pl in players:
        x1, y1, x2, y2 = map(int, pl.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_PLAYER, 2)
        label = f"P{pl.track_id}"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_PLAYER, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(pl.foot_x), int(pl.foot_y)), 4, C_PLAYER, -1)
    return frame


def draw_ball(frame: np.ndarray, ball: Optional[BallObservation], trail: deque) -> np.ndarray:
    # Estela con desvanecido
    pts = list(trail)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        a = i / max(len(pts), 1)
        cv2.line(frame, pts[i - 1], pts[i],
                 (0, int(80 + 175 * a), int(255 * a)), 2, cv2.LINE_AA)
    if ball is not None and ball.visible and ball.x is not None:
        c = (int(ball.x), int(ball.y))
        if getattr(ball, "interpolated", False):
            # posición rellenada: círculo hueco
            cv2.circle(frame, c, 6, C_BALL, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, c, 6, C_BALL, -1, cv2.LINE_AA)
            cv2.circle(frame, c, 8, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def build_minimap(court_model, mm_len_px: int = 320, margin: int = 14):
    """Crea el mini-mapa cenital base de la pista y la función metro->minimap px.

    Orientación: eje largo (length) horizontal, ancho (width) vertical, igual que
    la vista de lado del vídeo."""
    L = court_model.length
    W = court_model.width
    tram = getattr(court_model, "tramline", 0.0)
    scale = mm_len_px / L
    w = int(mm_len_px + 2 * margin)
    h = int(W * scale + 2 * margin)

    def to_mm(cx, cy):
        return (int(margin + cy * scale), int(margin + cx * scale))

    base = np.full((h, w, 3), 30, np.uint8)            # fondo oscuro
    col = (180, 180, 180)
    # marco de dobles
    cv2.rectangle(base, to_mm(0, 0), to_mm(W, L), col, 1)
    # líneas de individuales
    cv2.line(base, to_mm(tram, 0), to_mm(tram, L), col, 1)
    cv2.line(base, to_mm(W - tram, 0), to_mm(W - tram, L), col, 1)
    # red
    cv2.line(base, to_mm(0, L / 2), to_mm(W, L / 2), (255, 255, 255), 1)
    # líneas de saque
    cv2.line(base, to_mm(tram, L / 2 - 6.40), to_mm(W - tram, L / 2 - 6.40), col, 1)
    cv2.line(base, to_mm(tram, L / 2 + 6.40), to_mm(W - tram, L / 2 + 6.40), col, 1)
    # central de saque
    cv2.line(base, to_mm(W / 2, L / 2 - 6.40), to_mm(W / 2, L / 2 + 6.40), col, 1)
    return {"base": base, "to_mm": to_mm, "w": w, "h": h}


def draw_minimap(frame, mm, bounces, frame_idx, shots=None):
    """Superpone el mini-mapa (esquina sup. der.) con los botes acumulados y un
    contador de botes y golpes hasta el frame actual."""
    canvas = mm["base"].copy()
    h_mm, w_mm = canvas.shape[:2]
    n = 0
    for b in bounces:
        if b.frame > frame_idx:
            continue
        if b.court_x is None or not np.isfinite(b.court_x) or not np.isfinite(b.court_y):
            continue
        x, y = mm["to_mm"](b.court_x, b.court_y)
        x = int(min(max(x, 2), w_mm - 3))      # recortar al borde del mini-mapa
        y = int(min(max(y, 2), h_mm - 3))
        col = C_BOUNCE_IN if b.inside else C_BOUNCE_OUT
        cv2.circle(canvas, (x, y), 4, col, -1, cv2.LINE_AA)
        n += 1
    past = [s for s in (shots or []) if s.frame <= frame_idx]
    j1 = sum(1 for s in past if s.player_id == 1)
    j2 = sum(1 for s in past if s.player_id == 2)
    # Contadores (banda inferior, 2 líneas)
    cv2.rectangle(canvas, (0, h_mm - 38), (w_mm, h_mm), (0, 0, 0), -1)
    cv2.putText(canvas, f"Botes: {n}", (6, h_mm - 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BOUNCE_IN, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Golpes: {len(past)}  (J1:{j1}  J2:{j2})", (6, h_mm - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    H, W = frame.shape[:2]
    x0, y0 = W - mm["w"] - 10, 10
    if x0 < 0 or y0 + mm["h"] > H:
        return frame
    roi = frame[y0:y0 + mm["h"], x0:x0 + mm["w"]]
    cv2.addWeighted(canvas, 0.85, roi, 0.15, 0, roi)
    cv2.rectangle(frame, (x0, y0), (x0 + mm["w"], y0 + mm["h"]), (255, 255, 255), 1)
    return frame


def _legend(frame: np.ndarray) -> np.ndarray:
    items = [("Pista", C_COURT), ("Jugadores", C_PLAYER), ("Pelota", C_BALL)]
    x0, y0 = 10, 24
    cv2.rectangle(frame, (5, 5), (190, 18 + 26 * len(items)), (0, 0, 0), -1)
    for i, (txt, col) in enumerate(items):
        y = y0 + i * 26
        cv2.circle(frame, (x0 + 6, y - 5), 6, col, -1)
        cv2.putText(frame, txt, (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def render_annotated_video(
    input_path: str,
    out_path: str,
    fps: float,
    court_kps_by_frame: Dict[int, np.ndarray],
    players_by_frame: Dict[int, List[PlayerObservation]],
    ball_by_frame: Dict[int, BallObservation],
    stride: int = 1,
    max_frames: Optional[int] = None,
    trail_length: int = 40,
    max_trail_gap_px: float = 120.0,
    bounces: Optional[List[Bounce]] = None,
    draw_bounces: bool = False,
    draw_minimap_opt: bool = False,
    court_model=None,
    bounce_hold_frames: int = 20,
    minimap_width: int = 320,
    shots: Optional[list] = None,
) -> str:
    """Re-lee el vídeo y escribe una copia anotada con todas las detecciones."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    reader = VideoReader(input_path, stride, max_frames)
    W, Hpx = reader.meta.width, reader.meta.height
    bounces = bounces or []
    mm = (build_minimap(court_model, minimap_width)
          if (draw_minimap_opt and court_model is not None) else None)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, Hpx))
    if not writer.isOpened():
        reader.release()
        raise RuntimeError(f"No se pudo crear el vídeo de salida: {out_path}")

    # Keypoints de pista: usar la estimación válida más reciente (cámara fija).
    court_frames = sorted(court_kps_by_frame.keys())

    def court_at(idx: int) -> Optional[np.ndarray]:
        pos = bisect_right(court_frames, idx) - 1
        return court_kps_by_frame[court_frames[pos]] if pos >= 0 else None

    trail: deque = deque(maxlen=trail_length)
    from tqdm import tqdm

    n = 0
    for frame_idx, frame in tqdm(reader, desc="Vídeo anotado", unit="f"):
        draw_court(frame, court_at(frame_idx))
        draw_players(frame, players_by_frame.get(frame_idx, []))

        ball = ball_by_frame.get(frame_idx)
        if ball is not None and ball.visible and ball.x is not None:
            c = (int(ball.x), int(ball.y))
            if trail and trail[-1] is not None and \
                    np.hypot(c[0] - trail[-1][0], c[1] - trail[-1][1]) > max_trail_gap_px:
                trail.clear()
            trail.append(c)
        else:
            trail.append(None)
        draw_ball(frame, ball, trail)

        if draw_bounces and bounces:
            draw_bounce_markers(frame, bounces, frame_idx, bounce_hold_frames)
        if shots:
            draw_shot_markers(frame, shots, ball_by_frame, frame_idx, bounce_hold_frames)
        if mm is not None:
            draw_minimap(frame, mm, bounces, frame_idx, shots)

        _legend(frame)
        writer.write(frame)
        n += 1

    writer.release()
    reader.release()
    logger.info("Vídeo anotado escrito (%d frames) en: %s", n, out_path)
    return out_path
