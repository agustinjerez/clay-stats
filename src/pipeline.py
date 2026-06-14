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

import os
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
        crop_cfg = models["ball"].get("crop_to_court", {})
        tile_cfg = models["ball"].get("tile", {})
        cameras_data: Dict[str, dict] = {}
        for label, cam in cameras.items():
            path = cam["video"]
            meta, fps = self._video_meta(path)

            # ---- PISTA primero (necesaria para recorte y homografía) ----
            cframe = None
            if court_enabled:
                ccfg_cam = dict(ccfg)
                if cam.get("keypoints_json"):
                    ccfg_cam["keypoints_json"] = cam["keypoints_json"]
                court_det = build_court_detector(ccfg_cam)
                cframe = self._detect_court_once(court_det, path)

            # ---- Detección de pelota + jugadores ----
            ox = oy = 0
            if tile_cfg.get("enabled", False):
                # Tiling: parte el vídeo a lo ancho -> pelota más grande a igual imgsz
                with StepTimer(logger, f"[{label}] detección por tiles"):
                    ball_obs, players_by_frame = self._detect_tiled(
                        path, meta, combined, det, ball_det, player_det)
            else:
                det_path = path
                if crop_cfg.get("enabled", False) and cframe is not None and cframe.valid:
                    det_path, ox, oy = self._crop_to_court(
                        path, cframe.keypoints, meta, crop_cfg.get("margin_frac", 0.2), label)
                with StepTimer(logger, f"[{label}] detección pelota + jugadores"
                                        + (" (recorte de pista)" if (ox or oy) else "")):
                    ball_obs, players_by_frame = self._detect_source(
                        det_path, combined, det, ball_det, player_det)
                if ox or oy:                              # reposicionar a frame completo
                    self._offset_detections(ball_obs, players_by_frame, ox, oy)
                if det_path != path and os.path.exists(det_path):
                    os.remove(det_path)

            # ---- Homografía + filtros + refinado + proyección ----
            court_kps = {}
            hc = None
            if court_enabled and cframe is not None:
                with StepTimer(logger, f"[{label}] pista + proyección a metros"):
                    hc = CourtModel(length=cm["length"],
                                    singles_width=cm.get("width", 8.23),
                                    doubles_width=cm.get("doubles_width", 10.97))
                    cframe = hc.estimate_homography(cframe)
                    H = cframe.homography
                    if cframe.valid and H is not None:
                        court_kps = {cframe.frame: cframe.keypoints}
                        self._filter_ball_roi(ball_obs, cframe.keypoints, meta[0], meta[1])
                        self._filter_player_roi(players_by_frame, cframe.keypoints)
                        self._refine_ball(ball_obs, fps)
                        self._project(ball_obs, players_by_frame, hc, H)
                        logger.info("[%s] pista fijada y proyectada a metros.", label)
                    else:
                        logger.warning("[%s] pista no válida; sin métricas.", label)
                        hc = None
                        self._cap_players(players_by_frame)
                        self._refine_ball(ball_obs, fps)
            else:
                self._cap_players(players_by_frame)
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
                        minimap_width=ocfg.get("minimap_width", 320),
                        shots=data.get("shots"),
                    )
                report["_artifacts"]["annotated_video"][label] = out_path

            # Combinar las cámaras en un único vídeo lado a lado (izq | der)
            if not single and ocfg.get("combine_cameras", True):
                combined = self._combine_annotated(
                    report["_artifacts"]["annotated_video"], ocfg["annotated_video"])
                if combined:
                    report["_artifacts"]["annotated_video"]["combined"] = combined

        return report

    @staticmethod
    def _combine_annotated(paths_by_label: dict, base_out: str) -> str:
        """Une los vídeos anotados de las cámaras lado a lado (izq|der) en uno."""
        import subprocess
        # Orden: izquierda primero, derecha después.
        order = sorted([k for k in paths_by_label if k != "combined"],
                       key=lambda k: (0 if "left" in k.lower() or "izq" in k.lower()
                                      else 1 if "right" in k.lower() or "der" in k.lower()
                                      else 2, k))
        vids = [paths_by_label[k] for k in order]
        if len(vids) < 2:
            return ""
        out = base_out[:-4] + "_full.mp4" if base_out.endswith(".mp4") else base_out + "_full.mp4"
        inputs = []
        for v in vids:
            inputs += ["-i", v]
        cmd = ["ffmpeg", "-y", "-v", "error", *inputs,
               "-filter_complex", f"hstack=inputs={len(vids)}", out]
        try:
            subprocess.run(cmd, check=True)
            logger.info("Vídeo combinado (lado a lado) escrito en: %s", out)
            return out
        except Exception as e:
            logger.warning("No se pudo combinar los vídeos (ffmpeg): %s", e)
            return ""

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

    def _dump_track(self, label, ball_obs, bounces, shots):
        """Vuelca la trayectoria de la pelota + eventos a CSV para inspección."""
        import csv
        import os
        bframes = {b.frame for b in bounces}
        sframes = {s.frame for s in shots}
        out_dir = os.path.dirname(self.cfg["output"]["json_path"]) or "."
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"ball_track_{label}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "x", "y", "visible", "interpolated",
                        "court_x", "court_y", "is_bounce", "is_shot"])
            for b in ball_obs:
                w.writerow([b.frame,
                            round(b.x, 1) if b.x is not None else "",
                            round(b.y, 1) if b.y is not None else "",
                            int(b.visible), int(getattr(b, "interpolated", False)),
                            round(b.court_x, 2) if b.court_x is not None else "",
                            round(b.court_y, 2) if b.court_y is not None else "",
                            int(b.frame in bframes), int(b.frame in sframes)])
        logger.info("Trayectoria volcada en: %s", path)

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
            data["shots"] = shots          # para el contador del mini-mapa
            if self.cfg["output"].get("dump_track", False):
                self._dump_track(label, data["ball"], bounces, shots)
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
    def _video_meta(self, path):
        """(width, height, n_frames) y fps efectivo del vídeo."""
        reader = VideoReader(path, self.cfg["video"]["frame_stride"],
                             self.cfg["video"]["max_frames"])
        meta = (reader.meta.width, reader.meta.height, reader.meta.n_frames)
        fps = reader.meta.fps / max(1, self.cfg["video"]["frame_stride"])
        reader.release()
        logger.info("Vídeo %s: %dx%d @ %.1f fps | %d frames",
                    path, meta[0], meta[1], reader.meta.fps, meta[2])
        return meta, fps

    def _detect_source(self, det_path, combined, det, ball_det, player_det):
        """Detecta pelota + jugadores sobre `det_path` (vídeo completo o recorte)."""
        if combined:
            return det.detect_video(det_path)
        ball_obs = ball_det.detect_video(det_path)
        players_by_frame = player_det.detect_video(det_path)
        return ball_obs, players_by_frame

    def _crop_rect(self, path, x0, y0, cw, ch, name):
        """Recorta `path` al rectángulo (x0,y0,cw,ch) con ffmpeg. Devuelve la
        ruta del recorte o None si falla."""
        import subprocess
        out_dir = os.path.dirname(self.cfg["output"]["json_path"]) or "."
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"_crop_{name}.mp4")
        vf = f"crop=min(iw\\,{cw}):min(ih\\,{ch}):{x0}:{y0}"
        last = ""
        for codec in (["-c:v", "libx264", "-preset", "fast"],
                      ["-c:v", "mpeg4", "-q:v", "3"]):
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                                "-vf", vf, *codec, "-an", out],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return out
            last = r.stderr.strip()
        logger.warning("recorte ffmpeg falló (%s): %s", name, last)
        return None

    def _crop_to_court(self, path, keypoints, meta, margin_frac, label):
        """Recorta al rectángulo de la pista (+margen). Devuelve (ruta, x0, y0)."""
        import numpy as np
        W, H = meta[0], meta[1]
        kp = keypoints[np.isfinite(keypoints).all(axis=1)]
        if len(kp) < 4:
            return path, 0, 0
        x0 = max(0, int(kp[:, 0].min() - margin_frac * np.ptp(kp[:, 0])))
        y0 = max(0, int(kp[:, 1].min() - margin_frac * np.ptp(kp[:, 1])))
        x1 = min(W, int(kp[:, 0].max() + margin_frac * np.ptp(kp[:, 0])))
        y1 = min(H, int(kp[:, 1].max() + margin_frac * np.ptp(kp[:, 1])))
        cw = (x1 - x0) // 2 * 2; ch = (y1 - y0) // 2 * 2
        if cw < 64 or ch < 64 or (cw >= W and ch >= H):
            return path, 0, 0
        out = self._crop_rect(path, x0, y0, cw, ch, label)
        if out is None:
            return path, 0, 0
        logger.info("[%s] recorte de pista %dx%d (de %dx%d)", label, cw, ch, W, H)
        return out, x0, y0

    def _detect_tiled(self, path, meta, combined, det, ball_det, player_det):
        """Parte el vídeo en N tiles horizontales (con solapamiento), detecta en
        cada uno (la pelota gana píxeles a igual imgsz) y fusiona: pelota = mejor
        score por frame; jugadores = unión deduplicada."""
        from collections import defaultdict
        tcfg = self.cfg["models"]["ball"].get("tile", {})
        n = max(2, int(tcfg.get("n_tiles", 2)))
        overlap = float(tcfg.get("overlap_frac", 0.08))
        W, H, nf = meta
        tw = W / n
        ow = int(overlap * tw)
        ch = H // 2 * 2

        best = {}                              # frame -> (score, BallObservation)
        players = defaultdict(list)
        for t in range(n):
            x0 = max(0, int(round(t * tw - ow)))
            x1 = min(W, int(round((t + 1) * tw + ow)))
            cw = (x1 - x0) // 2 * 2
            tile_path = self._crop_rect(path, x0, 0, cw, ch, f"tile{t}")
            xoff = x0 if tile_path else 0
            src = tile_path or path
            tb, tp = self._detect_source(src, combined, det, ball_det, player_det)
            if tile_path and os.path.exists(tile_path):
                os.remove(tile_path)
            # offset a frame completo
            for b in tb:
                if b.x is not None:
                    b.x += xoff
            for fr, pls in tp.items():
                for pl in pls:
                    a, bb, c, d = pl.bbox
                    pl.bbox = (a + xoff, bb, c + xoff, d)
                    pl.foot_x += xoff
                players[fr].extend(pls)
            # pelota: mejor score por frame
            for b in tb:
                if b.visible and b.x is not None:
                    cur = best.get(b.frame)
                    if cur is None or b.score > cur[0]:
                        best[b.frame] = (b.score, b)
            logger.info("tile %d/%d [x %d-%d]: pelota %d, jugadores %d",
                        t + 1, n, x0, x1, sum(1 for b in tb if b.visible),
                        sum(len(v) for v in tp.values()))

        ball_obs = [best[f][1] if f in best else BallObservation(frame=f)
                    for f in range(nf)]
        self._dedupe_players(players)
        return ball_obs, dict(players)

    @staticmethod
    def _dedupe_players(players_by_frame, min_dist=60.0):
        """Quita jugadores duplicados (mismo jugador visto en dos tiles del
        solapamiento): NMS por punto de pies, conservando el de mayor score."""
        for fr, pls in players_by_frame.items():
            pls.sort(key=lambda p: p.score, reverse=True)
            kept = []
            for p in pls:
                if all((p.foot_x - q.foot_x) ** 2 + (p.foot_y - q.foot_y) ** 2
                       > min_dist ** 2 for q in kept):
                    kept.append(p)
            players_by_frame[fr] = kept

    @staticmethod
    def _offset_detections(ball_obs, players_by_frame, ox, oy):
        """Suma el offset del recorte para volver a coordenadas de frame completo."""
        for b in ball_obs:
            if b.x is not None:
                b.x += ox; b.y += oy
        for pls in players_by_frame.values():
            for pl in pls:
                x1, y1, x2, y2 = pl.bbox
                pl.bbox = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)

    @staticmethod
    def _offset_detections(ball_obs, players_by_frame, ox, oy):
        """Suma el offset del recorte para volver a coordenadas de frame completo."""
        for b in ball_obs:
            if b.x is not None:
                b.x += ox; b.y += oy
        for pls in players_by_frame.values():
            for pl in pls:
                x1, y1, x2, y2 = pl.bbox
                pl.bbox = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
                pl.foot_x += ox; pl.foot_y += oy

    def _cap_players(self, players_by_frame):
        """Recorta a max_players por frame (mayor score). Sin filtro de zona."""
        mp = self.cfg["models"]["player"].get("max_players", 2)
        for fr, pls in players_by_frame.items():
            if len(pls) > mp:
                pls.sort(key=lambda p: p.score, reverse=True)
                players_by_frame[fr] = pls[:mp]

    def _filter_player_roi(self, players_by_frame, keypoints) -> int:
        """Descarta jugadores cuyo punto de pies cae fuera del polígono de la
        pista (+ margen para los que sacan tras la línea de fondo) y recorta a
        max_players. Elimina espectadores/personas del fondo."""
        pcfg = self.cfg["models"]["player"].get("roi", {})
        mp = self.cfg["models"]["player"].get("max_players", 2)
        if not pcfg.get("enabled", True) or keypoints is None:
            self._cap_players(players_by_frame)
            return 0
        import numpy as np
        import cv2
        kp = keypoints[np.isfinite(keypoints).all(axis=1)].astype(np.float32)
        if len(kp) < 4:
            self._cap_players(players_by_frame)
            return 0
        hull = cv2.convexHull(kp)
        bbox_h = float(kp[:, 1].max() - kp[:, 1].min())
        margin_px = pcfg.get("margin_frac", 0.4) * bbox_h   # margen tras los fondos
        removed = 0
        for fr, pls in players_by_frame.items():
            inside = []
            for pl in pls:
                d = cv2.pointPolygonTest(hull, (float(pl.foot_x), float(pl.foot_y)), True)
                if d >= -margin_px:
                    inside.append(pl)
                else:
                    removed += 1
            inside.sort(key=lambda p: p.score, reverse=True)
            players_by_frame[fr] = inside[:mp]
        if removed:
            logger.info("ROI jugadores: descartadas %d detecciones fuera de pista", removed)
        return removed

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
