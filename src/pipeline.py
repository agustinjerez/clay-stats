"""Orquestador del pipeline de inferencia y análisis.

Por defecto analiza UN solo vídeo (`video.input_path`). Opcionalmente soporta
dos cámaras sincronizadas (`video.cameras: {left, right}`).

Estado actual:
  - Pelota y jugadores se detectan con SAM3 y se refina la trayectoria de la
    pelota.
  - La detección de PISTA está DESACTIVADA (en desarrollo, YOLO-pose). Sin
    homografía no hay coordenadas métricas, así que las estadísticas métricas
    (botes/golpes en metros, rallies, errores) quedan PREPARADAS pero inactivas.
    Por ahora se guardan las trayectorias en píxeles para proyectarlas cuando la
    pista esté lista.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .analysis import (
    BounceDetector, CourtModel, RallySegmenter, ShotDetector, StatisticsBuilder,
    refine_ball_track,
)
from .datatypes import BallObservation, CourtFrame, PlayerObservation
from .models import build_ball_detector, build_court_detector, build_player_detector
from .utils.video import VideoReader
from .utils.logging_utils import get_logger, StepTimer

logger = get_logger(__name__)


class TennisPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        cm = cfg["court_model"]
        self.court_model = CourtModel(
            length=cm["length"],
            singles_width=cm.get("width", 8.23),
            doubles_width=cm.get("doubles_width", 10.97),
        )

    # ------------------------------------------------------------------
    def _cameras(self, vcfg: dict) -> Dict[str, str]:
        """Devuelve {etiqueta: ruta} de las cámaras configuradas.

        Soporta `video.cameras: {near: ..., far: ...}` y, por compatibilidad,
        un único `video.input_path` (etiqueta 'cam1')."""
        if vcfg.get("input_path"):
            return {"video": vcfg["input_path"]}
        if vcfg.get("cameras"):
            return dict(vcfg["cameras"])
        raise ValueError("Configura video.input_path o video.cameras (left/right)")

    # ------------------------------------------------------------------
    def run(self) -> dict:
        vcfg = self.cfg["video"]
        models = self.cfg["models"]
        cameras = self._cameras(vcfg)
        logger.info("Cámaras: %s", {k: v for k, v in cameras.items()})

        # Construir los detectores UNA vez y reutilizarlos en cada cámara.
        combined = (
            models.get("sam3_combined", False)
            and models["ball"].get("backend", "sam3").startswith("sam3")
            and models["player"].get("backend", "sam3") == "sam3"
        )
        det = ball_det = player_det = None
        if combined:
            from .models import Sam3CombinedDetector
            det = Sam3CombinedDetector.from_config(models)
        else:
            ball_det = build_ball_detector(models["ball"])
            player_det = build_player_detector(models["player"])

        # ---- PISTA (opcional) ----
        ccfg = models.get("court", {})
        court_enabled = ccfg.get("enabled", False)
        court_det = build_court_detector(ccfg) if court_enabled else None
        if not court_enabled:
            logger.warning("Detección de PISTA desactivada (court.enabled=false). "
                           "Se generará informe de detecciones en píxeles.")

        cm = self.cfg["court_model"]
        cameras_data: Dict[str, dict] = {}
        for label, path in cameras.items():
            with StepTimer(logger, f"[{label}] detección pelota + jugadores"):
                ball_obs, players_by_frame, fps, meta = self._detect_camera(
                    path, models, combined, det, ball_det, player_det)

            court_kps = {}
            hc = None
            if court_enabled:
                with StepTimer(logger, f"[{label}] pista + proyección a metros"):
                    cframe = self._detect_court_once(court_det, path)
                    hc = CourtModel(length=cm["length"],
                                    singles_width=cm.get("width", 8.23),
                                    doubles_width=cm.get("doubles_width", 10.97))
                    cframe = hc.estimate_homography(cframe)
                    H = cframe.homography
                    if cframe.valid and H is not None:
                        court_kps = {cframe.frame: cframe.keypoints}
                        # Filtrar falsos positivos de pelota fuera de la pista
                        self._filter_ball_roi(ball_obs, cframe.keypoints, meta[0], meta[1])
                        self._refine_ball(ball_obs, fps)
                        self._project(ball_obs, players_by_frame, hc, H)
                        logger.info("[%s] pista fijada y proyectada a metros.", label)
                    else:
                        logger.warning("[%s] pista no válida; sin métricas.", label)
                        hc = None
                        self._refine_ball(ball_obs, fps)
            else:
                self._refine_ball(ball_obs, fps)

            cameras_data[label] = {
                "path": path, "fps": fps,
                "width": meta[0], "height": meta[1], "n_frames": meta[2],
                "ball": ball_obs, "players": players_by_frame,
                "court_kps": court_kps, "hc": hc,
            }

        ocfg = self.cfg["output"]
        stats = StatisticsBuilder(self.court_model, fps,
                                  match_id=ocfg["match_id"])

        # ---- Estadísticas ----
        with StepTimer(logger, "Estadísticas"):
            usable = {k: v for k, v in cameras_data.items() if v["hc"] is not None}
            if usable:
                report = self._build_metric_report(usable, stats, ocfg["match_id"])
            else:
                report = stats.build_detection_report(cameras_data)
            stats.save_json(report, ocfg["json_path"])
            logger.info("JSON escrito en: %s", ocfg["json_path"])

        report["_artifacts"] = {"json": ocfg["json_path"], "annotated_video": {}}

        # ---- Vídeo anotado (pelota + jugadores + pista si la hay) ----
        if ocfg.get("annotated_video"):
            from .utils.visualization import render_annotated_video
            single = len(cameras_data) == 1
            for label, data in cameras_data.items():
                out_path = (ocfg["annotated_video"] if single
                            else self._annotated_path(ocfg["annotated_video"], label))
                with StepTimer(logger, f"[{label}] render de vídeo anotado"):
                    render_annotated_video(
                        input_path=data["path"], out_path=out_path, fps=data["fps"],
                        court_kps_by_frame=data["court_kps"],
                        players_by_frame=data["players"],
                        ball_by_frame={b.frame: b for b in data["ball"]},
                        stride=vcfg["frame_stride"], max_frames=vcfg["max_frames"],
                        bounces=data.get("bounces"),
                        draw_bounces=ocfg.get("draw_bounces", False),
                        draw_minimap_opt=ocfg.get("draw_minimap", False),
                        court_model=data.get("hc"),
                        bounce_hold_frames=ocfg.get("bounce_hold_frames", 20),
                        minimap_width=ocfg.get("minimap_width", 220),
                    )
                report["_artifacts"]["annotated_video"][label] = out_path

        return report

    # ------------------------------------------------------------------
    def _project(self, ball_obs, players_by_frame, hc, H):
        """Proyecta pelota y pies de jugadores a coordenadas de pista (metros)."""
        for b in ball_obs:
            if b.visible and b.x is not None:
                proj = hc.to_court(b.x, b.y, H)
                if proj:
                    b.court_x, b.court_y = proj
        for frame_players in players_by_frame.values():
            for pl in frame_players:
                proj = hc.to_court(pl.foot_x, pl.foot_y, H)
                if proj:
                    pl.court_x, pl.court_y = proj

    def _analyze(self, ball_obs, players_by_frame, hc, fps):
        """Botes, golpes y rallies sobre una fuente con homografía válida."""
        acfg = self.cfg["analysis"]
        bounces = BounceDetector(
            hc, min_prominence_px=acfg["bounce"]["min_prominence_px"],
            smooth_window=acfg["bounce"]["smooth_window"],
            out_margin_m=acfg["error"]["out_margin_m"]).detect(ball_obs)
        shot_det = ShotDetector(
            hc, fps,
            min_frames_between_shots=acfg["shot"]["min_frames_between_shots"],
            smooth_window=acfg["shot"].get("smooth_window", 7),
            max_gap_frames=acfg["shot"].get("max_gap_frames", 12),
            prominence_frac=acfg["shot"].get("prominence_frac", 0.12),
            min_prominence_px=acfg["shot"].get("min_prominence_px", 25))
        player_sides = ShotDetector.player_sides(players_by_frame, hc)
        shots = shot_det.detect(ball_obs, players_by_frame, player_sides)
        rally_gap = int(round(acfg["rally"].get("max_gap_s", 2.5) * fps))
        rallies = RallySegmenter(hc, rally_gap).segment(shots, bounces)
        logger.info("Botes=%d, golpes=%d, rallies=%d", len(bounces), len(shots), len(rallies))
        return bounces, shots, rallies, player_sides

    def _build_metric_report(self, usable, stats, match_id) -> dict:
        """Informe métrico (botes/golpes/rallies/estadísticas) por fuente."""
        sources = {}
        for label, data in usable.items():
            hc, fps = data["hc"], data["fps"]
            bounces, shots, rallies, player_sides = self._analyze(
                data["ball"], data["players"], hc, fps)
            data["bounces"] = bounces      # para dibujarlos en el vídeo
            player_ids = [player_sides["left"], player_sides["right"]]
            src_stats = StatisticsBuilder(hc, fps, match_id=match_id)
            sources[label] = src_stats.build(player_ids, shots, bounces, rallies,
                                             data["players"], player_sides=player_sides)
        if len(sources) == 1:                 # un solo vídeo -> informe directo
            rep = next(iter(sources.values()))
            rep["court_available"] = True
            return rep
        from datetime import datetime, timezone
        return {"match_id": match_id, "court_available": True,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sources": sources}

    # ------------------------------------------------------------------
    def _detect_camera(self, path, models, combined, det, ball_det, player_det):
        """Detección de pelota + jugadores en una cámara y refinado de pelota."""
        reader = VideoReader(path, self.cfg["video"]["frame_stride"],
                             self.cfg["video"]["max_frames"])
        meta = (reader.meta.width, reader.meta.height, reader.meta.n_frames)
        fps = reader.meta.fps / max(1, self.cfg["video"]["frame_stride"])
        reader.release()
        logger.info("Vídeo %s: %dx%d @ %.1f fps | %d frames",
                    path, meta[0], meta[1], reader.meta.fps, meta[2])

        if combined:
            ball_obs, players_by_frame = det.detect_video(path)
        else:
            ball_obs = ball_det.detect_video(path)
            players_by_frame = player_det.detect_video(path)
        return ball_obs, players_by_frame, fps, meta

    def _refine_ball(self, ball_obs, fps):
        rcfg = self.cfg["models"]["ball"].get("refine", {})
        if rcfg.get("enabled", True):
            refine_ball_track(
                ball_obs, fps,
                max_interp_gap=rcfg.get("max_interp_gap", 8),
                hampel_window=rcfg.get("hampel_window", 7),
                hampel_sigma=rcfg.get("hampel_sigma", 3.0),
                max_speed_px_s=rcfg.get("max_speed_px_s", 4000.0),
                smooth_window=rcfg.get("smooth_window", 7),
                smooth_poly=rcfg.get("smooth_poly", 2),
            )

    def _filter_ball_roi(self, ball_obs, keypoints, w, h) -> int:
        """Descarta detecciones de pelota fuera de la zona de pista (+ margen).

        Mata falsos positivos lejos de la pista (vallas, fondo, reflejos)."""
        roicfg = self.cfg["models"]["ball"].get("roi", {})
        if not roicfg.get("enabled", True) or keypoints is None:
            return 0
        import numpy as np
        kp = keypoints[np.isfinite(keypoints).all(axis=1)]
        if len(kp) < 4:
            return 0
        x0, y0 = kp[:, 0].min(), kp[:, 1].min()
        x1, y1 = kp[:, 0].max(), kp[:, 1].max()
        bw, bh = x1 - x0, y1 - y0
        mx = roicfg.get("margin_x_frac", 0.08) * bw
        mtop = roicfg.get("margin_top_frac", 0.6) * bh     # lobs: mucho margen arriba
        mbot = roicfg.get("margin_bottom_frac", 0.15) * bh
        x0 -= mx; x1 += mx; y0 -= mtop; y1 += mbot
        removed = 0
        for b in ball_obs:
            if b.visible and b.x is not None and not (x0 <= b.x <= x1 and y0 <= b.y <= y1):
                b.visible = False
                b.x = b.y = None
                removed += 1
        if removed:
            logger.info("ROI de pista: descartadas %d detecciones de pelota fuera de zona",
                        removed)
        return removed

    def _detect_court_once(self, court_det, path) -> CourtFrame:
        """Detecta la pista en el primer frame (cámara fija). Para el modo
        'fixed' cualquier frame vale; para 'yolo' prueba varios hasta validar."""
        ccfg = self.cfg["models"]["court"]
        max_try = ccfg.get("detect_max_frames", 30)
        reader = VideoReader(path, stride=self.cfg["video"]["frame_stride"],
                             max_frames=max_try)
        result = None
        for frame_idx, frame in reader:
            result = court_det.detect(frame, frame_idx)
            if result.valid:
                break
        reader.release()
        return result

    @staticmethod
    def _annotated_path(base: str, label: str) -> str:
        """Inserta la etiqueta de cámara en la ruta del vídeo anotado."""
        if base.endswith(".mp4"):
            return base[:-4] + f"_{label}.mp4"
        return f"{base}_{label}.mp4"
