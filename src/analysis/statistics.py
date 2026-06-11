"""Construcción de estadísticas del partido y export a JSON."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from ..datatypes import Bounce, PlayerObservation, Rally, Shot
from ..utils.geometry import euclidean
from .court import CourtModel


class StatisticsBuilder:
    """
    Calcula las estadísticas solicitadas y las serializa en un JSON listo
    para subir a un servicio web.

    Estadísticas principales:
      - Nº total de golpes por jugador
      - Mapa de botes de la pelota en pista
      - Rally de golpes más largo por jugador
      - Errores por jugador

    Extras:
      - Velocidad media/máx de pelota por jugador
      - Distancia recorrida por jugador
      - Zonas de bote (cuadrícula) y % dentro/fuera
    """

    def __init__(self, court: CourtModel, fps: float, match_id: str = "match"):
        self.court = court
        self.fps = fps
        self.match_id = match_id

    # ------------------------------------------------------------------
    def build_detection_report(self, cameras_data: Dict[str, dict]) -> dict:
        """Informe de DETECCIONES por cámara (sin pista/homografía todavía).

        Guarda las trayectorias en píxeles de pelota y jugadores de cada cámara,
        listas para proyectarse a metros cuando la detección de pista esté
        integrada. Las estadísticas métricas quedan en `stats_metricas: null`.
        """
        cameras = {}
        for label, data in cameras_data.items():
            ball = data["ball"]
            players_by_frame = data["players"]

            n = len(ball)
            n_vis = sum(1 for b in ball if b.visible)
            n_interp = sum(1 for b in ball if getattr(b, "interpolated", False))
            ball_points = [
                {"frame": b.frame, "x": round(b.x, 1), "y": round(b.y, 1),
                 "score": round(b.score, 3), "interpolated": bool(b.interpolated)}
                for b in ball if b.visible and b.x is not None
            ]

            # Jugadores agrupados por track_id (punto de pies en píxeles)
            tracks: Dict[int, list] = defaultdict(list)
            for frame in sorted(players_by_frame):
                for pl in players_by_frame[frame]:
                    tracks[pl.track_id].append({
                        "frame": pl.frame,
                        "foot_x": round(pl.foot_x, 1), "foot_y": round(pl.foot_y, 1),
                        "bbox": [round(v, 1) for v in pl.bbox],
                        "score": round(pl.score, 3),
                    })
            n_player_dets = sum(len(v) for v in tracks.values())

            cameras[label] = {
                "video": data["path"],
                "fps": round(data["fps"], 3),
                "width": data["width"], "height": data["height"],
                "n_frames": data["n_frames"],
                "ball": {
                    "frames_total": n,
                    "visible": n_vis,
                    "interpolated": n_interp,
                    "visible_pct": round(100 * n_vis / max(1, n), 1),
                    "track_px": ball_points,
                },
                "players": {
                    "num_tracks": len(tracks),
                    "num_detections": n_player_dets,
                    "tracks_px": {str(tid): pts for tid, pts in tracks.items()},
                },
            }

        return {
            "match_id": self.match_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "court_available": False,
            "note": ("Detección de pista en desarrollo: trayectorias en píxeles "
                     "por cámara (sincronizadas). Estadísticas métricas pendientes "
                     "de la homografía de pista."),
            "cameras": cameras,
            "stats_metricas": None,
        }

    # ------------------------------------------------------------------
    def build(
        self,
        player_ids: List[int],
        shots: List[Shot],
        bounces: List[Bounce],
        rallies: List[Rally],
        players_by_frame: Dict[int, List[PlayerObservation]],
        player_sides: Dict[str, int] = None,
    ) -> dict:
        # id -> lado ('near'/'far'); la distancia se calcula por lado (robusto a
        # los cambios de track_id de SAM3).
        side_of_id = {}
        if player_sides:
            side_of_id = {v: k for k, v in player_sides.items()}

        per_player = {
            pid: self._player_block(pid, shots, rallies, players_by_frame,
                                    side_of_id.get(pid))
            for pid in player_ids
        }

        report = {
            "match_id": self.match_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fps": self.fps,
            "court": {"length_m": self.court.length, "width_m": self.court.width},
            "summary": {
                "total_shots": len(shots),
                "total_bounces": len(bounces),
                "total_rallies": len(rallies),
                "longest_rally_shots": max((len(r.shots) for r in rallies), default=0),
            },
            "players": per_player,
            "bounce_map": self._bounce_map(bounces),
            "rallies": [self._rally_block(r) for r in rallies],
        }
        return report

    # ------------------------------------------------------------------
    def _player_block(self, pid: int, shots, rallies, players_by_frame,
                      side: str = None) -> dict:
        p_shots = [s for s in shots if s.player_id == pid]

        # Rally más largo en el que participa este jugador
        longest = 0
        for r in rallies:
            if any(s.player_id == pid for s in r.shots):
                longest = max(longest, len(r.shots))

        errors = sum(1 for r in rallies if r.error_player == pid)

        speeds = [s.ball_speed_kmh for s in p_shots if s.ball_speed_kmh is not None]

        return {
            "player_id": pid,
            "side": side,
            "total_shots": len(p_shots),
            "longest_rally_shots": longest,
            "errors": errors,
            "avg_shot_speed_kmh": round(float(np.mean(speeds)), 1) if speeds else None,
            "max_shot_speed_kmh": round(float(np.max(speeds)), 1) if speeds else None,
            "distance_covered_m": self._distance(pid, players_by_frame, side),
        }

    def _distance(self, pid: int, players_by_frame, side: str = None) -> float:
        """Distancia recorrida (m). Si se da `side`, se asigna cada detección al
        lado por su court_y (robusto a cambios de track_id); si no, por track_id."""
        mid = self.court.length / 2
        track = []
        for frame in sorted(players_by_frame):
            chosen = None
            for pl in players_by_frame[frame]:
                if pl.court_x is None or pl.court_y is None:
                    continue
                if side is not None:
                    pl_side = "near" if pl.court_y < mid else "far"
                    if pl_side != side:
                        continue
                    if chosen is None or pl.score > chosen.score:
                        chosen = pl
                elif pl.track_id == pid:
                    chosen = pl
            if chosen is not None:
                track.append((chosen.court_x, chosen.court_y))
        dist = sum(euclidean(a, b) for a, b in zip(track, track[1:]))
        return round(float(dist), 1)

    # ------------------------------------------------------------------
    def _bounce_map(self, bounces: List[Bounce]) -> dict:
        points = [
            {
                "frame": b.frame,
                "x_m": round(b.court_x, 2),
                "y_m": round(b.court_y, 2),
                "inside": bool(b.inside),
                "side": b.side,
            }
            for b in bounces
        ]
        inside = sum(1 for b in bounces if b.inside)
        # Cuadrícula 3x3 (ancho x largo) para zonas de bote
        grid = defaultdict(int)
        for b in bounces:
            cx = min(2, max(0, int(b.court_x / (self.court.width / 3))))
            cy = min(2, max(0, int(b.court_y / (self.court.length / 3))))
            grid[f"{cx}_{cy}"] += 1
        return {
            "points": points,
            "total": len(bounces),
            "inside": inside,
            "outside": len(bounces) - inside,
            "zones_3x3": dict(grid),
        }

    def _rally_block(self, r: Rally) -> dict:
        return {
            "start_frame": r.start_frame,
            "end_frame": r.end_frame,
            "num_shots": len(r.shots),
            "ended_by": r.ended_by,
            "error_player": r.error_player,
        }

    # ------------------------------------------------------------------
    def save_json(self, report: dict, path: str) -> str:
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return path

    def save_heatmap(self, bounces: List[Bounce], path: str):
        """Heatmap PNG de los botes sobre un esquema de pista."""
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        L, W = self.court.length, self.court.width
        fig, ax = plt.subplots(figsize=(4, 9))

        # Esquema de pista (marco de dobles + líneas de individuales)
        tram = getattr(self.court, "tramline", 0.0)
        ax.add_patch(plt.Rectangle((0, 0), W, L, fill=False, lw=2))   # dobles
        ax.plot([tram, tram], [0, L], "k-", lw=1)                     # individual izq
        ax.plot([W - tram, W - tram], [0, L], "k-", lw=1)             # individual der
        ax.plot([0, W], [L / 2, L / 2], "k-", lw=2)                   # red
        ax.plot([tram, W - tram], [L / 2 - 6.40, L / 2 - 6.40], "k--", lw=1)
        ax.plot([tram, W - tram], [L / 2 + 6.40, L / 2 + 6.40], "k--", lw=1)
        ax.plot([W / 2, W / 2], [L / 2 - 6.40, L / 2 + 6.40], "k--", lw=1)

        xs = [b.court_x for b in bounces]
        ys = [b.court_y for b in bounces]
        if xs:
            ax.scatter(xs, ys, c="red", s=40, alpha=0.6, edgecolors="k")
        ax.set_xlim(-1, W + 1)
        ax.set_ylim(-1, L + 1)
        ax.set_aspect("equal")
        ax.set_title("Mapa de botes")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
