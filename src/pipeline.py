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
    def _cameras(self, vcfg: dict) -> Dict[str, dict]:
        """Devuelve {etiqueta: {video, keypoints_json}} de las cámaras.

        Soporta:
          video.input_path: ruta            (una cámara)
          video.cameras:
            left:  data/cam1.mp4            (cadena -> calibración global)
            right:
              video: data/cam2.mp4         (dict -> calibración propia)
              keypoints_json: weights/court_keypoints_cam2.json
        """
        default_kp = self.cfg.get("models", {}).get("court", {}).get("keypoints_json")
        if vcfg.get("input_path"):
            return {"video": {"video": vcfg["input_path"], "keypoints_json": default_kp}}
        if vcfg.get("cameras"):
            out = {}
            for label, val in vcfg["cameras"].items():
                if isinstance(val, dict):
                    out[label] = {"video": val["video"],
                                  "keypoints_json": val.get("keypoints_json", default_kp)}
                else:
                    out[label] = {"video": val, "keypoints_json": default_kp}
            return out
        raise ValueError("Configura video.input_path o video.cameras (left/right)")

    # ------------------------------------------------------------------
    def run(self) -> dict:
        vcfg = self.cfg["video"]
        models = self.cfg["models"]
        cameras = self._cameras(vcfg)
        logger.info("Cámaras: %s", {k: v["video"] for k, v in cameras.items()})

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

        # ---- PISTA (opcional, una calibración POR cámara) ----
        ccfg = models.get("court", {})
        court_enabled = ccfg.get("enabled", False)
        if not court_enabled:
            logger.warning("Detección de PISTA desactivada (court.enabled=false). "
                           "Se generará informe de detecciones en píxeles.")

        cm = self.cfg["court_model"]
        cameras_data: Dict[str, dict] = {}
        for label, cam in cameras.items():
            path = cam["video"]
            with StepTimer(logger, f"[{label}] detección pelota + jugadores"):
                ball_obs, players_by_frame, fps, meta = self._detect_camera(
                    path, models, combined, det, ball_det, player_det)

            court_kps = {}
            hc = None
            if court_enabled:
                with StepTimer(logger, f"[{label}] pista + proyección a metros"):
                    # Calibración propia de esta cámara
                    ccfg_cam = dict(ccfg)
                    if cam.get("keypoints_json"):
                        ccfg_cam["keypoints_json"] = cam["keypoints_json"]
                    court_det = build_court_detector(ccfg_cam)
                    logger.info("[%s] pista desde %s", label,
                                ccfg_cam.get("keypoints_json", ccfg_cam.get("weights")))
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
        """Informe métrico (botes/golpes/rallies/estadísticas) por fuente, y si
        hay 2+ cámaras, un informe de PARTIDO fusionado."""
        sources = {}
        analyzed = {}
        for label, data in usable.items():
            hc, fps = data["hc"], data["fps"]
            bounces, shots, rallies, player_sides = self._analyze(
                data["ball"], data["players"], hc, fps)
            data["bounces"] = bounces      # para dibujarlos en el vídeo
            analyzed[label] = {"bounces": bounces, "shots": shots,
                               "players": data["players"], "fps": fps}
            player_ids = [player_sides["left"], player_sides["right"]]
            src_stats = StatisticsBuilder(hc, fps, match_id=match_id)
            sources[label] = src_stats.build(player_ids, shots, bounces, rallies,
                                             data["players"], player_sides=player_sides)
        if len(sources) == 1:                 # un solo vídeo -> informe directo
            rep = next(iter(sources.values()))
            rep["court_available"] = True
            return rep
        # Dos o más cámaras -> fusionar en un único informe de partido.
        report = self._fuse(analyzed, match_id)
        report["by_camera"] = sources
        return report

    def _fuse(self, analyzed: dict, match_id: str) -> dict:
        """Fusiona las cámaras (mismo sistema métrico, sincronizadas) en un
        único informe de partido: cada cámara aporta los golpes de SU jugador;
        botes y golpes se combinan por frame y se re-segmentan los rallies."""
        from dataclasses import replace
        from collections import defaultdict
        from .analysis.court import LEFT_PLAYER_ID, RIGHT_PLAYER_ID

        labels = list(analyzed.keys())

        def pid_for(label, i):
            lo = label.lower()
            if "left" in lo or "izq" in lo:
                return LEFT_PLAYER_ID
            if "right" in lo or "der" in lo:
                return RIGHT_PLAYER_ID
            return LEFT_PLAYER_ID if i == 0 else RIGHT_PLAYER_ID

        all_shots, all_bounces = [], []
        merged_players = defaultdict(list)
        fps = next(iter(analyzed.values()))["fps"]
        for i, (label, a) in enumerate(analyzed.items()):
            pid = pid_for(label, i)
            all_shots.extend(replace(s, player_id=pid) for s in a["shots"])
            all_bounces.extend(a["bounces"])
            for fr, pls in a["players"].items():
                merged_players[fr].extend(pls)

        all_shots.sort(key=lambda s: s.frame)
        all_bounces.sort(key=lambda b: b.frame)

        court = CourtModel(length=self.cfg["court_model"]["length"],
                           singles_width=self.cfg["court_model"].get("width", 8.23),
                           doubles_width=self.cfg["court_model"].get("doubles_width", 10.97))
        rally_gap = int(round(self.cfg["analysis"]["rally"].get("max_gap_s", 2.5) * fps))
        rallies = RallySegmenter(court, rally_gap).segment(all_shots, all_bounces)
        player_sides = {"left": LEFT_PLAYER_ID, "right": RIGHT_PLAYER_ID}

        stats = StatisticsBuilder(court, fps, match_id=match_id)
        report = stats.build([LEFT_PLAYER_ID, RIGHT_PLAYER_ID], all_shots, all_bounces,
                             rallies, merged_players, player_sides=player_sides)
        report["court_available"] = True
        report["fused_from"] = labels
        logger.info("Partido fusionado: %d golpes, %d botes, %d rallies (cámaras %s)",
                    len(all_shots), len(all_bounces), len(rallies), labels)
        return report

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
        """Descarta detecciones de pelota fuera del POLÍGONO de la pista (+margen).

        Usa la envolvente convexa de los keypoints marcados (la pista en
        perspectiva es un trapecio, no un rectángulo) y un margen pequeño. Mata
        falsos positivos fuera de la pista (vallas, fondo, reflejos)."""
        roicfg = self.cfg["models"]["ball"].get("roi", {})
        if not roicfg.get("enabled", True) or keypoints is None:
            return 0
        import numpy as np
        import cv2
        kp = keypoints[np.isfinite(keypoints).all(axis=1)].astype(np.float32)
        if len(kp) < 4:
            return 0
        hull = cv2.convexHull(kp)
        bbox_h = float(kp[:, 1].max() - kp[:, 1].min())
        # margen permitido fuera del polígono (px). Sube margin_frac para lobs.
        margin_px = roicfg.get("margin_frac", 0.05) * bbox_h
        removed = 0
        for b in ball_obs:
            if not (b.visible and b.x is not None):
                continue
            dist = cv2.pointPolygonTest(hull, (float(b.x), float(b.y)), True)
            if dist < -margin_px:           # fuera del polígono más allá del margen
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
