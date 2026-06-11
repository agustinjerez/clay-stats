from .court_detector import (
    CourtDetector, CourtYoloDetector, FixedCourtDetector, build_court_detector,
)
from .ball_detector import (
    BallDetector, Sam3VideoBallDetector, Sam3ImageBallDetector, Sam3CombinedDetector,
)
from .player_detector import PlayerDetector, Sam3PlayerDetector
from .tracknet import TrackNetBallDetector, TrackNetV2


def build_player_detector(cfg: dict):
    """Factory de detector de jugadores. backend: "sam3" (def.) | "yolo"."""
    backend = cfg.get("backend", "sam3").lower()
    if backend == "sam3":
        return Sam3PlayerDetector.from_config(cfg)
    if backend == "yolo":
        return PlayerDetector.from_config(cfg)
    raise ValueError(f"Backend de jugadores desconocido: {backend!r}")


def build_ball_detector(cfg: dict):
    """Factory: elige el backend de detección de pelota según la config.

    backend:
      "sam3" | "sam3_video" -> Sam3VideoBallDetector (tracking nativo de vídeo)
      "sam3_image"          -> Sam3ImageBallDetector (per-frame, notebook)
      "tracknet"            -> TrackNetBallDetector
    Todos exponen `detect_video(video_path) -> List[BallObservation]`.
    """
    backend = cfg.get("backend", "sam3").lower()
    if backend in ("sam3", "sam3_video"):
        return Sam3VideoBallDetector.from_config(cfg)
    if backend == "sam3_image":
        return Sam3ImageBallDetector.from_config(cfg)
    if backend == "tracknet":
        return TrackNetBallDetector.from_config(cfg)
    raise ValueError(f"Backend de pelota desconocido: {backend!r}")


__all__ = [
    "CourtDetector",
    "CourtYoloDetector",
    "FixedCourtDetector",
    "build_court_detector",
    "BallDetector",
    "Sam3VideoBallDetector",
    "Sam3ImageBallDetector",
    "Sam3CombinedDetector",
    "PlayerDetector",
    "Sam3PlayerDetector",
    "TrackNetBallDetector",
    "TrackNetV2",
    "build_ball_detector",
    "build_player_detector",
]
