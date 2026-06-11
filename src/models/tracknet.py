"""TrackNetV2: detección de la pelota de tenis mediante heatmaps.

Arquitectura encoder-decoder estilo VGG16 que recibe N frames consecutivos
apilados (N*3 canales RGB) y produce N heatmaps (uno por frame) con la
posición de la pelota como una gaussiana. Es el estándar de facto para
seguimiento de objetos pequeños y rápidos (pelota/volante) en deportes.

Referencia: Sun et al., "TrackNetV2: Efficient Shuttlecock Tracking Network".

El wrapper `TrackNetBallDetector` expone la MISMA interfaz pública que el
detector basado en SAM3 (`detect_video`), de modo que el pipeline no cambia.
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..datatypes import BallObservation
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
#  Arquitectura
# ----------------------------------------------------------------------
def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c),
    )


class TrackNetV2(nn.Module):
    """
    in_frames: nº de frames consecutivos de entrada (3 por defecto).
    Entrada : (B, in_frames*3, H, W)
    Salida  : (B, in_frames, H, W) con valores en [0, 1] (sigmoid).
    """

    def __init__(self, in_frames: int = 3):
        super().__init__()
        in_c = in_frames * 3
        self.in_frames = in_frames

        # Encoder (VGG16-like)
        self.e1 = nn.Sequential(_conv_block(in_c, 64), _conv_block(64, 64))
        self.e2 = nn.Sequential(_conv_block(64, 128), _conv_block(128, 128))
        self.e3 = nn.Sequential(
            _conv_block(128, 256), _conv_block(256, 256), _conv_block(256, 256)
        )
        self.e4 = nn.Sequential(
            _conv_block(256, 512), _conv_block(512, 512), _conv_block(512, 512)
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Decoder
        self.d3 = nn.Sequential(
            _conv_block(768, 256), _conv_block(256, 256), _conv_block(256, 256)
        )
        self.d2 = nn.Sequential(_conv_block(384, 128), _conv_block(128, 128))
        self.d1 = nn.Sequential(_conv_block(192, 64), _conv_block(64, 64))
        self.out = nn.Conv2d(64, in_frames, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e1(x)                 # H
        x2 = self.e2(self.pool(x1))     # H/2
        x3 = self.e3(self.pool(x2))     # H/4
        x4 = self.e4(self.pool(x3))     # H/8 (bottleneck)

        y = torch.cat([self.up(x4), x3], dim=1)   # H/4
        y = self.d3(y)
        y = torch.cat([self.up(y), x2], dim=1)    # H/2
        y = self.d2(y)
        y = torch.cat([self.up(y), x1], dim=1)    # H
        y = self.d1(y)
        return torch.sigmoid(self.out(y))


# ----------------------------------------------------------------------
#  Wrapper de inferencia
# ----------------------------------------------------------------------
class TrackNetBallDetector:
    def __init__(
        self,
        weights: Optional[str] = None,
        in_frames: int = 3,
        input_size=(288, 512),     # (alto, ancho) de la red
        device: str = "cuda",
        heatmap_thresh: float = 0.5,
    ):
        from ..utils.device import resolve_device
        self.in_frames = in_frames
        self.h, self.w = input_size
        self.thresh = heatmap_thresh
        self.device = resolve_device(device)

        logger.info("TrackNet: inicializando (device=%s, in_frames=%d, input=%dx%d)",
                    self.device, in_frames, self.w, self.h)
        self.model = TrackNetV2(in_frames=in_frames).to(self.device).eval()
        if weights and os.path.exists(weights):
            ckpt = torch.load(weights, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state, strict=False)
            logger.info("TrackNet: pesos cargados desde %s", weights)
        else:
            logger.warning(
                "TrackNet: sin pesos válidos (%s) -> inicialización aleatoria "
                "(solo valida el pipeline; no detectará bien la pelota).", weights
            )

    # ------------------------------------------------------------------
    def _preprocess(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        chans = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.w, self.h)).astype(np.float32) / 255.0
            chans.append(rgb)
        stacked = np.concatenate(chans, axis=2)            # H, W, in_frames*3
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _heatmap_to_point(self, hm: np.ndarray, sx: float, sy: float):
        """Centroide del mayor blob del heatmap, escalado a la imagen original."""
        mask = (hm >= self.thresh).astype(np.uint8)
        if mask.sum() == 0:
            return None
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cx, cy = centroids[largest]
        area = stats[largest, cv2.CC_STAT_AREA] / mask.size
        return cx * sx, cy * sy, float(area)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def detect_video(self, video_path: str) -> List[BallObservation]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el vídeo: {video_path}")
        W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sx, sy = W0 / self.w, H0 / self.h
        logger.info("TrackNet: procesando %s (%dx%d, %d frames)",
                    video_path, W0, H0, n_total)

        observations: List[BallObservation] = []
        buffer: List[np.ndarray] = []
        base_idx = 0   # índice del primer frame del buffer
        pbar = tqdm(total=n_total or None, desc="Pelota (TrackNet)", unit="f")

        def flush(frames):
            nonlocal base_idx
            tensor = self._preprocess(frames)
            heatmaps = self.model(tensor)[0].cpu().numpy()   # (in_frames, h, w)
            for k in range(self.in_frames):
                obs = BallObservation(frame=base_idx + k)
                pt = self._heatmap_to_point(heatmaps[k], sx, sy)
                if pt is not None:
                    obs.x, obs.y, obs.score = pt
                    obs.visible = True
                observations.append(obs)
            base_idx += self.in_frames

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buffer.append(frame)
            idx += 1
            pbar.update(1)
            if len(buffer) == self.in_frames:
                flush(buffer)
                buffer = []
        # Resto (< in_frames): rellenar repitiendo el último frame
        if buffer:
            while len(buffer) < self.in_frames:
                buffer.append(buffer[-1])
            flush(buffer)
        pbar.close()
        cap.release()
        n_vis = sum(1 for o in observations if o.visible)
        logger.info("TrackNet: %d observaciones, pelota visible en %d (%.1f%%)",
                    len(observations), n_vis, 100 * n_vis / max(1, len(observations)))
        return observations

    # ------------------------------------------------------------------
    @staticmethod
    def from_config(cfg: dict) -> "TrackNetBallDetector":
        return TrackNetBallDetector(
            weights=cfg.get("weights"),
            in_frames=cfg.get("in_frames", 3),
            input_size=tuple(cfg.get("input_size", (288, 512))),
            device=cfg.get("device", "cuda"),
            heatmap_thresh=cfg.get("heatmap_thresh", 0.5),
        )
