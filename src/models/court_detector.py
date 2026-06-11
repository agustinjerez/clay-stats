"""Detección de la pista con YOLOv8-pose (sustituye al detector TrackNet).

La pista se graba con DOS cámaras desde el mismo trípode, en el centro del lado
largo (a la altura de la red). Cada cámara ve una MEDIA pista (red -> fondo).
Un modelo YOLO-pose detecta los 13 keypoints de la media pista (ver
`src/analysis/court.py: COURT_KEYPOINT_NAMES`), con los que se calcula la
homografía imagen->metros para situar botes (pelota) y golpes (jugadores).

No hay un modelo YOLO de pista 'de fábrica': hay que entrenarlo con tus frames.
Ver `training/court_yolo/` (etiquetado + entrenamiento).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..datatypes import CourtFrame
from ..analysis.court import COURT_NUM_KP
from ..utils.device import resolve_device
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class CourtYoloDetector:
    """Wrapper de inferencia YOLOv8-pose para los keypoints de media pista."""

    def __init__(self, weights: str, device: str = "auto",
                 conf: float = 0.30, kpt_conf: float = 0.30,
                 num_keypoints: int = COURT_NUM_KP, imgsz: int = 1280):
        from ultralytics import YOLO

        self.device = resolve_device(device)
        self.conf = conf
        self.kpt_conf = kpt_conf
        self.num_keypoints = num_keypoints
        self.imgsz = imgsz
        logger.info("CourtYoloDetector: cargando %s (device=%s, kpts=%d)",
                    weights, self.device, num_keypoints)
        self.model = YOLO(weights)

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray, frame_idx: int = 0) -> CourtFrame:
        """Detecta la media pista en un frame y devuelve sus keypoints (px)."""
        res = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]

        kps = np.full((self.num_keypoints, 2), np.nan, dtype=np.float32)
        if res.keypoints is not None and len(res.keypoints) > 0:
            # Quedarse con la instancia de pista de mayor confianza
            idx = 0
            if res.boxes is not None and res.boxes.conf is not None and len(res.boxes) > 1:
                idx = int(res.boxes.conf.argmax().item())
            xy = res.keypoints.xy[idx].cpu().numpy()              # (K, 2)
            conf = (res.keypoints.conf[idx].cpu().numpy()
                    if res.keypoints.conf is not None else np.ones(len(xy)))
            k = min(self.num_keypoints, len(xy))
            for i in range(k):
                if conf[i] >= self.kpt_conf and (xy[i] > 0).all():
                    kps[i] = xy[i]

        n_ok = int(np.isfinite(kps).all(axis=1).sum())
        valid = n_ok >= 4
        logger.debug("Frame %d: %d/%d keypoints de pista (válido=%s)",
                     frame_idx, n_ok, self.num_keypoints, valid)
        return CourtFrame(frame=frame_idx, keypoints=kps, valid=valid)

    # ------------------------------------------------------------------
    @staticmethod
    def from_config(cfg: dict) -> "CourtYoloDetector":
        return CourtYoloDetector(
            weights=cfg["weights"],
            device=cfg.get("device", "auto"),
            conf=cfg.get("conf", 0.30),
            kpt_conf=cfg.get("kpt_conf", 0.30),
            num_keypoints=cfg.get("num_keypoints", COURT_NUM_KP),
            imgsz=cfg.get("imgsz", 1280),
        )


class FixedCourtDetector:
    """Pista FIJA a partir de una calibración de una sola imagen (JSON).

    Cámara fija -> los 13 keypoints son constantes. Se cargan de un JSON
    (generado con tools/label_court_once.py) y se reaplican a cada frame,
    escalándolos si la resolución del frame difiere de la de calibración.
    No necesita modelo ni GPU.
    """

    def __init__(self, keypoints_json: str):
        import json
        with open(keypoints_json, "r") as f:
            data = json.load(f)
        self.ref_w = int(data["width"])
        self.ref_h = int(data["height"])
        kps = data["keypoints"]
        self.keypoints_ref = np.array(
            [[k[0], k[1]] if k is not None else [np.nan, np.nan] for k in kps],
            dtype=np.float32,
        )
        self.num_keypoints = len(self.keypoints_ref)
        n_ok = int(np.isfinite(self.keypoints_ref).all(axis=1).sum())
        logger.info("FixedCourtDetector: %d/%d keypoints desde %s (calib %dx%d)",
                    n_ok, self.num_keypoints, keypoints_json, self.ref_w, self.ref_h)

    def detect(self, frame_bgr: np.ndarray, frame_idx: int = 0) -> CourtFrame:
        h, w = frame_bgr.shape[:2]
        sx, sy = w / self.ref_w, h / self.ref_h
        kps = self.keypoints_ref.copy()
        kps[:, 0] *= sx
        kps[:, 1] *= sy
        valid = int(np.isfinite(kps).all(axis=1).sum()) >= 4
        return CourtFrame(frame=frame_idx, keypoints=kps, valid=valid)

    @staticmethod
    def from_config(cfg: dict) -> "FixedCourtDetector":
        return FixedCourtDetector(keypoints_json=cfg["keypoints_json"])


def build_court_detector(cfg: dict):
    """Factory de detección de pista. mode: "fixed" (calibración 1 imagen) | "yolo"."""
    mode = cfg.get("mode", "fixed").lower()
    if mode == "fixed":
        return FixedCourtDetector.from_config(cfg)
    if mode == "yolo":
        return CourtYoloDetector.from_config(cfg)
    raise ValueError(f"Modo de pista desconocido: {mode!r}")


# Alias de compatibilidad con el nombre anterior.
CourtDetector = CourtYoloDetector
