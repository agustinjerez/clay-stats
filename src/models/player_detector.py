"""Detección y seguimiento de jugadores.

Dos backends, misma interfaz `detect_video(path) -> {frame: [PlayerObservation]}`:

  - `Sam3PlayerDetector` (por defecto): SAM3 con prompt de texto
    ("tennis player") vía `SAM3VideoSemanticPredictor`. Detecta y sigue a los
    jugadores con identidades estables y, a diferencia de YOLO, localiza bien al
    jugador del fondo (pequeño en la imagen).
  - `PlayerDetector` (YOLO): alternativa con `model.track`.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from ..datatypes import PlayerObservation
from ..utils.device import resolve_device
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _obs_from_boxes(frame_idx, boxes, ids, scores, max_players) -> List[PlayerObservation]:
    """Construye PlayerObservation desde cajas/ids/scores y se queda con los
    `max_players` de mayor confianza."""
    obs = []
    for i, (box, sc) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(float, box)
        tid = int(ids[i]) if ids is not None else i + 1
        obs.append(PlayerObservation(
            frame=frame_idx, track_id=tid,
            bbox=(x1, y1, x2, y2),
            foot_x=(x1 + x2) / 2.0, foot_y=y2,   # punto de contacto con el suelo
            score=float(sc),
        ))
    obs.sort(key=lambda o: o.score, reverse=True)
    return obs[:max_players]


# ======================================================================
#  SAM3 (por defecto)
# ======================================================================
class Sam3PlayerDetector:
    def __init__(self, weights: str, device: str = "cuda",
                 prompt: str = "tennis player", score_threshold: float = 0.30,
                 imgsz: int = 640, max_players: int = 2, half=None):
        self.prompt = prompt
        self.score_threshold = score_threshold
        self.imgsz = imgsz
        self.max_players = max_players
        # Pool de candidatos antes del filtro de zona (el recorte final a
        # max_players lo hace el pipeline tras descartar los de fuera de pista).
        self.candidate_max = max(6, 3 * max_players)
        self.device = resolve_device(device)
        self.half = self.device.startswith("cuda") if half is None else half
        self._predictor = self._load(weights)

    def _load(self, weights):
        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
        except ImportError as e:
            raise ImportError("SAM3VideoSemanticPredictor no disponible; "
                              "pip install -U 'ultralytics>=8.3.237'") from e
        overrides = dict(conf=self.score_threshold, task="segment", mode="predict",
                         imgsz=self.imgsz, model=weights, half=self.half,
                         save=False, verbose=False, device=self.device)
        logger.info("SAM3 jugadores: cargando %s (device=%s, prompt=%r)",
                    weights, self.device, self.prompt)
        return SAM3VideoSemanticPredictor(overrides=overrides)

    def detect_video(self, video_path: str) -> Dict[int, List[PlayerObservation]]:
        from tqdm import tqdm

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        logger.info("SAM3 jugadores: tracking nativo sobre %s (%d frames)",
                    video_path, total)

        results = self._predictor(source=video_path, text=[self.prompt], stream=True)
        out: Dict[int, List[PlayerObservation]] = {}
        for frame_idx, r in enumerate(tqdm(results, total=total or None,
                                           desc="Jugadores (SAM3)", unit="f")):
            obs = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                ids = (r.boxes.id.cpu().numpy() if getattr(r.boxes, "id", None) is not None
                       else None)
                obs = _obs_from_boxes(frame_idx, boxes, ids, scores, self.candidate_max)
            out[frame_idx] = obs
        n = sum(len(v) for v in out.values())
        logger.info("SAM3 jugadores: %d detecciones en %d frames (media %.2f/frame)",
                    n, len(out), n / max(1, len(out)))
        return out

    @staticmethod
    def from_config(cfg: dict) -> "Sam3PlayerDetector":
        return Sam3PlayerDetector(
            weights=cfg["weights"], device=cfg.get("device", "cuda"),
            prompt=cfg.get("prompt", "tennis player"),
            score_threshold=cfg.get("score_threshold", cfg.get("conf", 0.30)),
            imgsz=cfg.get("imgsz", 640), max_players=cfg.get("max_players", 2),
            half=cfg.get("half", None),
        )


# ======================================================================
#  YOLO (alternativa)
# ======================================================================
class PlayerDetector:
    def __init__(self, weights: str, device: str = "cuda", conf: float = 0.35,
                 iou: float = 0.5, person_class_id: int = 0, max_players: int = 2):
        from ultralytics import YOLO

        self.device = resolve_device(device)
        logger.info("YOLO jugadores: cargando %s (device=%s, conf=%.2f)",
                    weights, self.device, conf)
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.person_class_id = person_class_id
        self.max_players = max_players
        self.candidate_max = max(6, 3 * max_players)

    def detect_video(self, video_path: str) -> Dict[int, List[PlayerObservation]]:
        results = self.model.track(
            source=video_path, stream=True, persist=True,
            conf=self.conf, iou=self.iou, classes=[self.person_class_id],
            device=self.device, verbose=False,
        )
        out: Dict[int, List[PlayerObservation]] = {}
        for frame_idx, r in enumerate(results):
            obs = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                ids = (r.boxes.id.cpu().numpy() if r.boxes.id is not None else None)
                obs = _obs_from_boxes(frame_idx, boxes, ids, scores, self.candidate_max)
            out[frame_idx] = obs
        return out

    @staticmethod
    def from_config(cfg: dict) -> "PlayerDetector":
        return PlayerDetector(
            weights=cfg["weights"], device=cfg.get("device", "cuda"),
            conf=cfg.get("conf", 0.35), iou=cfg.get("iou", 0.5),
            person_class_id=cfg.get("person_class_id", 0),
            max_players=cfg.get("max_players", 2),
        )
