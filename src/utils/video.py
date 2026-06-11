"""Lectura y escritura de vídeo con OpenCV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    n_frames: int


class VideoReader:
    """Iterador de frames con stride y límite opcional."""

    def __init__(self, path: str, stride: int = 1, max_frames: Optional[int] = None):
        self.path = path
        self.stride = max(1, int(stride))
        self.max_frames = max_frames
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el vídeo: {path}")
        self.meta = VideoMeta(
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            n_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        idx = 0
        emitted = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if idx % self.stride == 0:
                yield idx, frame
                emitted += 1
                if self.max_frames and emitted >= self.max_frames:
                    break
            idx += 1
        self.cap.release()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    """Escritor de vídeo anotado."""

    def __init__(self, path: str, fps: float, size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
