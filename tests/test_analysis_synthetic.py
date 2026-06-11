"""Test de humo de la capa de análisis con trayectorias sintéticas.

No requiere torch / SAM3 / YOLO: valida geometría, botes, golpes, rallies,
errores y export JSON con datos simulados.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import (
    BounceDetector, CourtModel, RallySegmenter, ShotDetector, StatisticsBuilder,
)
from src.datatypes import BallObservation, PlayerObservation


def build_synthetic():
    """Vista de lado: la pelota va y viene en X (imagen) entre los dos
    jugadores (izquierda/derecha), con un bote a media pista por trayecto."""
    court = CourtModel(length=23.77, singles_width=8.23, doubles_width=10.97)
    fps = 30.0
    balls, players_by_frame = [], {}

    # Jugadores: izquierda (court_y bajo, x imagen baja) y derecha (alto).
    pL = (4.0, 2.0)
    pR = (4.0, 21.0)

    def img_x(cy):           # eje largo (court_y) -> X de imagen
        return 100 + cy * 50

    frame = 0
    waypoints = [pL, pR, pL, pR, pL]   # rally de 4: reversiones en X
    for seg in range(len(waypoints) - 1):
        a, b = np.array(waypoints[seg]), np.array(waypoints[seg + 1])
        for t in np.linspace(0, 1, 12, endpoint=False):
            pos = a + (b - a) * t
            img_y = 300 - 100 * np.sin(np.pi * t)   # parábola = bote
            obs = BallObservation(frame=frame, x=img_x(pos[1]), y=img_y,
                                  court_x=float(pos[0]), court_y=float(pos[1]),
                                  score=1.0, visible=True)
            balls.append(obs)
            players_by_frame[frame] = [
                PlayerObservation(frame, 1, (0, 0, 1, 1), img_x(pL[1]), 500,
                                  court_x=pL[0], court_y=pL[1], score=0.9),
                PlayerObservation(frame, 2, (0, 0, 1, 1), img_x(pR[1]), 500,
                                  court_x=pR[0], court_y=pR[1], score=0.9),
            ]
            frame += 1
    return court, fps, balls, players_by_frame


def test_pipeline_analysis():
    court, fps, balls, pbf = build_synthetic()

    bounces = BounceDetector(court, min_prominence_px=20).detect(balls)
    shots = ShotDetector(court, fps, min_frames_between_shots=3,
                         smooth_window=7, prominence_frac=0.2,
                         min_prominence_px=25).detect(balls, pbf)
    rallies = RallySegmenter(court, max_gap_frames=25).segment(shots, bounces)

    stats = StatisticsBuilder(court, fps, match_id="test")
    report = stats.build([1, 2], shots, bounces, rallies, pbf)

    print("Botes:", len(bounces))
    print("Golpes:", len(shots), [(s.player_id, s.frame) for s in shots])
    print("Rallies:", [(r.start_frame, r.end_frame, len(r.shots), r.ended_by)
                       for r in rallies])
    print("Jugadores:", report["players"])

    assert len(bounces) > 0, "deberían detectarse botes"
    assert len(shots) > 0, "deberían detectarse golpes"
    assert report["summary"]["total_rallies"] >= 1
    assert set(report["players"].keys()) == {1, 2}

    out = stats.save_json(report, "/tmp/test_stats.json")
    assert os.path.exists(out)
    print("JSON OK ->", out)
    print("\nTODO OK")


if __name__ == "__main__":
    test_pipeline_analysis()
