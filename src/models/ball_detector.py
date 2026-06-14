"""Detección y seguimiento de la pelota con SAM3 (Meta) vía Ultralytics.

Dos modos, ambos con la misma interfaz pública `detect_video(path) -> [BallObservation]`:

1. `Sam3VideoBallDetector` (RECOMENDADO) — tracking NATIVO de vídeo con
   `SAM3VideoSemanticPredictor`. SAM3 detecta y SIGUE el concepto a lo largo del
   vídeo con memoria temporal e identidades estables (no re-detecta desde cero
   cada frame). La doc de Ultralytics recomienda esto frente a la inferencia
   frame a frame para no reasignar IDs en cada frame.

2. `Sam3ImageBallDetector` — segmenta cada frame por separado con
   `SAM3SemanticPredictor` + tracker propio por movimiento. Portado del notebook
   `sam3_tennis_ball_tracker.py` (puntos 5–7). Útil como alternativa/depuración.

La clasificación de botes/golpes NO se hace aquí: la realiza la capa de análisis
del proyecto en coordenadas métricas.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..datatypes import BallObservation
from ..utils.device import resolve_device
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ======================================================================
#  Mixin con los filtros geométricos y el tracker por movimiento
# ======================================================================
class _BallCandidateMixin:
    score_threshold: float
    min_box_px: int
    max_box_rel: float
    aspect_ratio_range: tuple
    max_ball_speed_px_s: float
    pred_weight: float
    velocity_ema_alpha: float
    max_gap_frames: int

    def _candidate_list(self, result, max_side_px: float) -> List[dict]:
        """Filtra las cajas candidatas por tamaño y aspect ratio."""
        if result.boxes is None or len(result.boxes) == 0:
            return []
        return self._filter_candidates(result.boxes.xyxy.cpu().numpy(),
                                       result.boxes.conf.cpu().numpy(), max_side_px)

    def _filter_candidates(self, boxes, confs, max_side_px: float) -> List[dict]:
        cands = []
        ar_lo, ar_hi = self.aspect_ratio_range
        for box, conf in zip(boxes, confs):
            if conf < self.score_threshold:
                continue
            bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
            if bw < self.min_box_px or bh < self.min_box_px:
                continue
            if bw > max_side_px or bh > max_side_px:
                continue
            ar = bw / bh if bh > 0 else 99
            if not (ar_lo <= ar <= ar_hi):
                continue
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            cands.append({"box": box, "score": float(conf),
                          "center": np.array([cx, cy], dtype=np.float64)})
        return cands

    def _select_candidate(self, cands, prev_center, prev_t, prev_velocity, t, fps):
        """Elige el mejor candidato con un tracker por movimiento."""
        if not cands:
            return None
        if prev_center is None:
            return max(cands, key=lambda c: c["score"])
        dt = max(t - prev_t, 1.0 / fps)
        pred = prev_center + prev_velocity * dt
        max_disp = self.max_ball_speed_px_s * dt + 30.0
        feas = [(c, float(np.linalg.norm(c["center"] - pred))) for c in cands]
        feas = [(c, d) for c, d in feas if d <= max_disp]
        if not feas:
            return None
        return max(feas, key=lambda cd: cd[0]["score"]
                   - self.pred_weight * (cd[1] / max_disp))[0]


# ======================================================================
#  1) Tracking NATIVO de vídeo (SAM3VideoSemanticPredictor)
# ======================================================================
class Sam3VideoBallDetector(_BallCandidateMixin):
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        prompt: str = "small yellow tennis ball",
        score_threshold: float = 0.30,
        imgsz: int = 960,
        min_box_px: int = 3,
        max_box_rel: float = 0.05,
        aspect_ratio_range=(0.4, 2.5),
        max_ball_speed_px_s: float = 2500,
        pred_weight: float = 0.4,
        velocity_ema_alpha: float = 0.6,
        max_gap_frames: int = 8,
        half: Optional[bool] = None,
    ):
        self.weights = weights
        self.prompt = prompt
        self.score_threshold = score_threshold
        self.imgsz = imgsz
        self.min_box_px = min_box_px
        self.max_box_rel = max_box_rel
        self.aspect_ratio_range = tuple(aspect_ratio_range)
        self.max_ball_speed_px_s = max_ball_speed_px_s
        self.pred_weight = pred_weight
        self.velocity_ema_alpha = velocity_ema_alpha
        self.max_gap_frames = max_gap_frames
        self.device = resolve_device(device)
        self.half = self.device.startswith("cuda") if half is None else half
        self._predictor = self._load(weights)

    def _load(self, weights):
        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
        except ImportError as e:
            raise ImportError(
                "SAM3VideoSemanticPredictor no disponible. Necesitas "
                "ultralytics>=8.3.237: pip install -U ultralytics"
            ) from e
        overrides = dict(conf=self.score_threshold, task="segment", mode="predict",
                         imgsz=self.imgsz, model=weights, half=self.half,
                         save=False, verbose=False, device=self.device)
        logger.info("SAM3 (vídeo): cargando %s (device=%s, half=%s, prompt=%r)",
                    weights, self.device, self.half, self.prompt)
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        logger.info("SAM3 (vídeo): predictor listo")
        return predictor

    def detect_video(self, video_path: str) -> List[BallObservation]:
        from tqdm import tqdm

        cap = cv2.VideoCapture(video_path)
        fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if cap.isOpened() else 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        logger.info("SAM3 (vídeo): tracking nativo sobre %s (%d frames @ %.1f fps)",
                    video_path, total, fps)

        # Stream de resultados: un Results por frame, con IDs de tracking estables.
        results = self._predictor(source=video_path, text=[self.prompt], stream=True)

        observations: List[BallObservation] = []
        prev_center, prev_t = None, None
        prev_velocity = np.array([0.0, 0.0])
        gap_frames = 0

        for frame_idx, r in enumerate(tqdm(results, total=total or None,
                                           desc="Pelota (SAM3 vídeo)", unit="f")):
            h0, w0 = r.orig_shape if getattr(r, "orig_shape", None) else (0, 0)
            max_side_px = self.max_box_rel * (w0 or 1280)
            t = frame_idx / fps if fps > 0 else frame_idx

            cands = self._candidate_list(r, max_side_px)
            chosen = self._select_candidate(cands, prev_center, prev_t,
                                            prev_velocity, t, fps)

            obs = BallObservation(frame=frame_idx)
            if chosen is not None:
                center = chosen["center"]
                if prev_center is not None and prev_t is not None and (t - prev_t) > 0:
                    v = (center - prev_center) / (t - prev_t)
                    prev_velocity = (self.velocity_ema_alpha * v
                                     + (1 - self.velocity_ema_alpha) * prev_velocity)
                obs.x, obs.y = float(center[0]), float(center[1])
                obs.score = chosen["score"]
                obs.visible = True
                prev_center, prev_t, gap_frames = center, t, 0
            else:
                gap_frames += 1
                if gap_frames > self.max_gap_frames:
                    prev_center, prev_t = None, None
                    prev_velocity = np.array([0.0, 0.0])
            observations.append(obs)

        n_vis = sum(1 for o in observations if o.visible)
        logger.info("SAM3 (vídeo): pelota detectada en %d/%d frames (%.1f%%)",
                    n_vis, len(observations), 100 * n_vis / max(1, len(observations)))
        return observations

    @staticmethod
    def from_config(cfg: dict) -> "Sam3VideoBallDetector":
        return Sam3VideoBallDetector(
            weights=cfg["weights"],
            device=cfg.get("device", "cuda"),
            prompt=cfg.get("prompt", "small yellow tennis ball"),
            score_threshold=cfg.get("score_threshold", 0.30),
            imgsz=cfg.get("imgsz", 960),
            min_box_px=cfg.get("min_box_px", 3),
            max_box_rel=cfg.get("max_box_rel", 0.05),
            aspect_ratio_range=cfg.get("aspect_ratio_range", (0.4, 2.5)),
            max_ball_speed_px_s=cfg.get("max_ball_speed_px_s", 2500),
            pred_weight=cfg.get("pred_weight", 0.4),
            velocity_ema_alpha=cfg.get("velocity_ema_alpha", 0.6),
            max_gap_frames=cfg.get("max_gap_frames", 8),
            half=cfg.get("half", None),
        )


# ======================================================================
#  2) Per-frame (SAM3SemanticPredictor) — portado del notebook
# ======================================================================
class Sam3ImageBallDetector(_BallCandidateMixin):
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        prompt: str = "small yellow tennis ball",
        score_threshold: float = 0.30,
        min_box_px: int = 3,
        max_box_rel: float = 0.05,
        aspect_ratio_range=(0.4, 2.5),
        max_ball_speed_px_s: float = 2500,
        pred_weight: float = 0.4,
        velocity_ema_alpha: float = 0.6,
        max_gap_frames: int = 8,
        half: Optional[bool] = None,
    ):
        self.prompt = prompt
        self.score_threshold = score_threshold
        self.min_box_px = min_box_px
        self.max_box_rel = max_box_rel
        self.aspect_ratio_range = tuple(aspect_ratio_range)
        self.max_ball_speed_px_s = max_ball_speed_px_s
        self.pred_weight = pred_weight
        self.velocity_ema_alpha = velocity_ema_alpha
        self.max_gap_frames = max_gap_frames
        self.device = resolve_device(device)
        self.half = self.device.startswith("cuda") if half is None else half
        self._predictor = self._load(weights)

    def _load(self, weights):
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError as e:
            raise ImportError(
                "SAM3SemanticPredictor no disponible. Necesitas "
                "ultralytics>=8.3.237: pip install -U ultralytics"
            ) from e
        overrides = dict(conf=self.score_threshold, task="segment", mode="predict",
                         model=weights, half=self.half, save=False,
                         verbose=False, device=self.device)
        logger.info("SAM3 (imagen): cargando %s (device=%s, half=%s, prompt=%r)",
                    weights, self.device, self.half, self.prompt)
        predictor = SAM3SemanticPredictor(overrides=overrides)
        logger.info("SAM3 (imagen): predictor listo")
        return predictor

    def detect_video(self, video_path: str) -> List[BallObservation]:
        from tqdm import tqdm

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el vídeo: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        Hpx = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_side_px = self.max_box_rel * W
        logger.info("SAM3 (imagen): procesando %s (%dx%d @ %.1f fps, %d frames)",
                    video_path, W, Hpx, fps, total)

        prev_center, prev_t = None, None
        prev_velocity = np.array([0.0, 0.0])
        gap_frames = 0
        observations: List[BallObservation] = []

        for frame_idx in tqdm(range(total), desc="Pelota (SAM3 imagen)", unit="f"):
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_idx / fps if fps > 0 else frame_idx
            self._predictor.set_image(frame)
            r = self._predictor(text=[self.prompt])
            r = r[0] if isinstance(r, (list, tuple)) else r
            cands = self._candidate_list(r, max_side_px)
            chosen = self._select_candidate(cands, prev_center, prev_t,
                                            prev_velocity, t, fps)

            obs = BallObservation(frame=frame_idx)
            if chosen is not None:
                center = chosen["center"]
                if prev_center is not None and prev_t is not None and (t - prev_t) > 0:
                    v = (center - prev_center) / (t - prev_t)
                    prev_velocity = (self.velocity_ema_alpha * v
                                     + (1 - self.velocity_ema_alpha) * prev_velocity)
                obs.x, obs.y = float(center[0]), float(center[1])
                obs.score = chosen["score"]
                obs.visible = True
                prev_center, prev_t, gap_frames = center, t, 0
            else:
                gap_frames += 1
                if gap_frames > self.max_gap_frames:
                    prev_center, prev_t = None, None
                    prev_velocity = np.array([0.0, 0.0])
            observations.append(obs)

        cap.release()
        n_vis = sum(1 for o in observations if o.visible)
        logger.info("SAM3 (imagen): pelota detectada en %d/%d frames (%.1f%%)",
                    n_vis, len(observations), 100 * n_vis / max(1, len(observations)))
        return observations

    @staticmethod
    def from_config(cfg: dict) -> "Sam3ImageBallDetector":
        return Sam3ImageBallDetector(
            weights=cfg["weights"],
            device=cfg.get("device", "cuda"),
            prompt=cfg.get("prompt", "small yellow tennis ball"),
            score_threshold=cfg.get("score_threshold", 0.30),
            min_box_px=cfg.get("min_box_px", 3),
            max_box_rel=cfg.get("max_box_rel", 0.05),
            aspect_ratio_range=cfg.get("aspect_ratio_range", (0.4, 2.5)),
            max_ball_speed_px_s=cfg.get("max_ball_speed_px_s", 2500),
            pred_weight=cfg.get("pred_weight", 0.4),
            velocity_ema_alpha=cfg.get("velocity_ema_alpha", 0.6),
            max_gap_frames=cfg.get("max_gap_frames", 8),
            half=cfg.get("half", None),
        )


# ======================================================================
#  3) Pasada ÚNICA de SAM3: pelota + jugadores a la vez
# ======================================================================
class Sam3CombinedDetector(_BallCandidateMixin):
    """Una sola pasada de `SAM3VideoSemanticPredictor` con dos conceptos
    (`[ball_prompt, player_prompt]`). Separa los resultados por índice de clase
    (0=pelota, 1=jugador), así se ahorra recorrer el vídeo dos veces.

    `detect_video` devuelve `(ball_obs, players_by_frame)`.
    """

    def __init__(self, weights, device="cuda",
                 ball_prompt="small yellow tennis ball",
                 player_prompt="tennis player",
                 score_threshold=0.30, imgsz=960,
                 min_box_px=3, max_box_rel=0.05, aspect_ratio_range=(0.4, 2.5),
                 max_ball_speed_px_s=2500, pred_weight=0.4,
                 velocity_ema_alpha=0.6, max_gap_frames=8,
                 max_players=2, half=None):
        self.ball_prompt = ball_prompt
        self.player_prompt = player_prompt
        self.text = [ball_prompt, player_prompt]      # idx 0=pelota, 1=jugador
        self.score_threshold = score_threshold
        self.imgsz = imgsz
        self.min_box_px = min_box_px
        self.max_box_rel = max_box_rel
        self.aspect_ratio_range = tuple(aspect_ratio_range)
        self.max_ball_speed_px_s = max_ball_speed_px_s
        self.pred_weight = pred_weight
        self.velocity_ema_alpha = velocity_ema_alpha
        self.max_gap_frames = max_gap_frames
        self.max_players = max_players
        self.candidate_max = max(6, 3 * max_players)   # pool antes del filtro de zona
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
        logger.info("SAM3 combinado: cargando %s (device=%s, prompts=%s)",
                    weights, self.device, self.text)
        return SAM3VideoSemanticPredictor(overrides=overrides)

    def detect_video(self, video_path: str):
        from tqdm import tqdm
        from .player_detector import _obs_from_boxes

        cap = cv2.VideoCapture(video_path)
        fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if cap.isOpened() else 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        logger.info("SAM3 combinado: una pasada sobre %s (%d frames @ %.1f fps)",
                    video_path, total, fps)

        results = self._predictor(source=video_path, text=self.text, stream=True)

        ball_obs: List[BallObservation] = []
        players_by_frame = {}
        prev_center, prev_t = None, None
        prev_velocity = np.array([0.0, 0.0])
        gap_frames = 0

        for frame_idx, r in enumerate(tqdm(results, total=total or None,
                                           desc="SAM3 (pelota+jugadores)", unit="f")):
            h0, w0 = r.orig_shape if getattr(r, "orig_shape", None) else (0, 1280)
            max_side_px = self.max_box_rel * (w0 or 1280)
            t = frame_idx / fps if fps > 0 else frame_idx

            # Separar por clase (0=pelota, 1=jugador)
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = (r.boxes.cls.cpu().numpy().astype(int)
                       if getattr(r.boxes, "cls", None) is not None
                       else np.zeros(len(xyxy), dtype=int))
                ids = (r.boxes.id.cpu().numpy() if getattr(r.boxes, "id", None) is not None
                       else None)
                bmask, pmask = cls == 0, cls == 1
            else:
                xyxy = np.empty((0, 4)); conf = np.empty((0,))
                cls = np.empty((0,), dtype=int); ids = None
                bmask = pmask = np.empty((0,), dtype=bool)

            # --- Pelota ---
            cands = self._filter_candidates(xyxy[bmask], conf[bmask], max_side_px) \
                if len(xyxy) else []
            chosen = self._select_candidate(cands, prev_center, prev_t,
                                            prev_velocity, t, fps)
            obs = BallObservation(frame=frame_idx)
            if chosen is not None:
                center = chosen["center"]
                if prev_center is not None and prev_t is not None and (t - prev_t) > 0:
                    v = (center - prev_center) / (t - prev_t)
                    prev_velocity = (self.velocity_ema_alpha * v
                                     + (1 - self.velocity_ema_alpha) * prev_velocity)
                obs.x, obs.y = float(center[0]), float(center[1])
                obs.score = chosen["score"]
                obs.visible = True
                prev_center, prev_t, gap_frames = center, t, 0
            else:
                gap_frames += 1
                if gap_frames > self.max_gap_frames:
                    prev_center, prev_t = None, None
                    prev_velocity = np.array([0.0, 0.0])
            ball_obs.append(obs)

            # --- Jugadores ---
            if len(xyxy) and pmask.any():
                pids = ids[pmask] if ids is not None else None
                players_by_frame[frame_idx] = _obs_from_boxes(
                    frame_idx, xyxy[pmask], pids, conf[pmask], self.candidate_max)
            else:
                players_by_frame[frame_idx] = []

        n_ball = sum(1 for o in ball_obs if o.visible)
        n_pl = sum(len(v) for v in players_by_frame.values())
        logger.info("SAM3 combinado: pelota en %d/%d frames; %d detecciones de jugador",
                    n_ball, len(ball_obs), n_pl)
        return ball_obs, players_by_frame

    @staticmethod
    def from_config(models_cfg: dict) -> "Sam3CombinedDetector":
        b = models_cfg["ball"]
        p = models_cfg["player"]
        return Sam3CombinedDetector(
            weights=b["weights"], device=b.get("device", "cuda"),
            ball_prompt=b.get("prompt", "small yellow tennis ball"),
            player_prompt=p.get("prompt", "tennis player"),
            score_threshold=min(b.get("score_threshold", 0.30),
                                p.get("score_threshold", p.get("conf", 0.30))),
            imgsz=b.get("imgsz", 960),  # combinado: usa el imgsz de la pelota
            min_box_px=b.get("min_box_px", 3),
            max_box_rel=b.get("max_box_rel", 0.05),
            aspect_ratio_range=b.get("aspect_ratio_range", (0.4, 2.5)),
            max_ball_speed_px_s=b.get("max_ball_speed_px_s", 2500),
            pred_weight=b.get("pred_weight", 0.4),
            velocity_ema_alpha=b.get("velocity_ema_alpha", 0.6),
            max_gap_frames=b.get("max_gap_frames", 8),
            max_players=p.get("max_players", 2),
            half=b.get("half", None),
        )


# Alias de compatibilidad (el backend "sam3" usa el tracking nativo de vídeo).
BallDetector = Sam3VideoBallDetector
