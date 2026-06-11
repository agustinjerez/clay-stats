"""Refinado de la trayectoria de la pelota (independiente del backend).

Mejora la calidad de las posiciones de la pelota antes del análisis:

  1. Filtro de Hampel sobre x e y: detecta y elimina falsos positivos (picos
     aislados que implican un salto imposible) sustituyéndolos por hueco.
  2. Rechazo por velocidad: descarta puntos cuya velocidad respecto al anterior
     supera un máximo físico (px/s).
  3. Interpolación de huecos CORTOS (<= max_interp_gap frames): la pelota suele
     perderse unos pocos frames; se rellena linealmente y se marca como
     `interpolated`.
  4. Suavizado Savitzky-Golay por tramos continuos.

Trabaja sobre las coordenadas de IMAGEN (x, y); la proyección a metros se hace
después, así el refinado beneficia tanto a botes como a golpes.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.signal import savgol_filter

from ..datatypes import BallObservation
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _hampel(arr: np.ndarray, window: int, n_sigma: float) -> np.ndarray:
    """Filtro de Hampel: sustituye outliers por NaN (no los rellena)."""
    out = arr.copy()
    n = len(arr)
    k = window // 2
    for i in range(n):
        if not np.isfinite(arr[i]):
            continue
        lo, hi = max(0, i - k), min(n, i + k + 1)
        win = arr[lo:hi]
        win = win[np.isfinite(win)]
        if len(win) < 3:
            continue
        med = np.median(win)
        mad = 1.4826 * np.median(np.abs(win - med))
        if mad > 0 and abs(arr[i] - med) > n_sigma * mad:
            out[i] = np.nan
    return out


def _reject_speed(x: np.ndarray, y: np.ndarray, fps: float,
                  max_speed_px_s: float) -> None:
    """Marca como NaN (in place) los puntos con velocidad imposible."""
    last_i = None
    for i in range(len(x)):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            continue
        if last_i is not None:
            dt = (i - last_i) / fps
            d = np.hypot(x[i] - x[last_i], y[i] - y[last_i])
            if dt > 0 and d / dt > max_speed_px_s:
                x[i] = np.nan
                y[i] = np.nan
                continue
        last_i = i


def _interp_short_gaps(arr: np.ndarray, max_gap: int):
    """Interpola linealmente sólo los huecos de longitud <= max_gap.

    Devuelve (array interpolado, máscara de posiciones rellenadas)."""
    out = arr.copy()
    filled = np.zeros(len(arr), dtype=bool)
    valid = np.where(np.isfinite(arr))[0]
    if len(valid) < 2:
        return out, filled
    for a, b in zip(valid, valid[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [arr[a], arr[b]])
            filled[a + 1:b] = True
    return out, filled


def _smooth_segments(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay aplicado a cada tramo continuo finito."""
    out = arr.copy()
    finite = np.isfinite(arr)
    i = 0
    n = len(arr)
    while i < n:
        if not finite[i]:
            i += 1
            continue
        j = i
        while j < n and finite[j]:
            j += 1
        seg = arr[i:j]
        if len(seg) >= 5:
            w = min(window, len(seg))
            if w % 2 == 0:
                w -= 1
            if w > poly:
                out[i:j] = savgol_filter(seg, w, poly)
        i = j
    return out


def refine_ball_track(
    balls: List[BallObservation],
    fps: float,
    max_interp_gap: int = 8,
    hampel_window: int = 7,
    hampel_sigma: float = 3.0,
    max_speed_px_s: float = 4000.0,
    smooth_window: int = 7,
    smooth_poly: int = 2,
) -> dict:
    """Refina in-place las posiciones de imagen de la pelota. Devuelve métricas."""
    n = len(balls)
    if n == 0:
        return {"raw": 0, "after": 0, "interpolated": 0, "removed": 0}

    x = np.array([b.x if (b.visible and b.x is not None) else np.nan for b in balls])
    y = np.array([b.y if (b.visible and b.y is not None) else np.nan for b in balls])
    raw = int(np.isfinite(x).sum())

    # 1) Hampel (outliers aislados)
    x = _hampel(x, hampel_window, hampel_sigma)
    y = _hampel(y, hampel_window, hampel_sigma)
    # 2) Velocidad imposible
    _reject_speed(x, y, fps, max_speed_px_s)
    removed = raw - int((np.isfinite(x)).sum())

    # 3) Interpolar huecos cortos
    xi, fx = _interp_short_gaps(x, max_interp_gap)
    yi, fy = _interp_short_gaps(y, max_interp_gap)
    filled = fx & fy
    # 4) Suavizado por tramos
    xs = _smooth_segments(xi, smooth_window, smooth_poly)
    ys = _smooth_segments(yi, smooth_window, smooth_poly)

    after = 0
    interpolated = 0
    for i, b in enumerate(balls):
        if np.isfinite(xs[i]) and np.isfinite(ys[i]):
            b.x, b.y = float(xs[i]), float(ys[i])
            b.visible = True
            b.interpolated = bool(filled[i])
            after += 1
            interpolated += int(filled[i])
        else:
            b.visible = False
            b.interpolated = False
            b.x = b.y = None

    stats = {"raw": raw, "after": after, "interpolated": interpolated, "removed": removed}
    logger.info("Refinado pelota: %d detecciones -> %d válidas (%d interpoladas, "
                "%d outliers eliminados)", raw, after, interpolated, removed)
    return stats
