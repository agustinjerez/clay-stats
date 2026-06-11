#!/usr/bin/env python3
"""
Validación visual de la detección de pista (pista completa, 14 keypoints).

Dibuja los 14 keypoints (con su índice) y el esqueleto de la pista sobre un
frame del vídeo o una imagen, para comprobar la calibración/modelo.

Uso:
    python infer_court.py --config config.yaml --video data/match.mp4 --frame 150 --draw-court
    python infer_court.py --config config.yaml --image data/images/frame_00000.jpg
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import yaml

from src.analysis.court import CourtModel
from src.models.court_detector import build_court_detector
from src.utils.logging_utils import setup_logging

# Aristas de la PISTA COMPLETA (19 keypoints). Longitudinales partidas en la red.
COURT_EDGES = [
    (0, 3), (10, 13), (4, 6), (7, 9),
    (14, 15), (15, 16), (16, 17), (17, 18),
    (0, 14), (14, 10), (3, 18), (18, 13),
    (1, 15), (15, 11), (2, 17), (17, 12),
    (5, 16), (16, 8),
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def grab_frame(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {args.image}")
        return img
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el vídeo: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {args.frame}")
    return frame


def draw_keypoints(img, kps):
    for i, (x, y) in enumerate(kps):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(img, p, 6, (0, 0, 255), -1)
        cv2.circle(img, p, 7, (255, 255, 255), 1)
        cv2.putText(img, str(i), (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_court_skeleton(img, kps):
    for a, b in COURT_EDGES:
        pa, pb = kps[a], kps[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     (0, 255, 0), 2, cv2.LINE_AA)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validación visual de keypoints de pista")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--frame", type=int, default=0, help="índice de frame (modo vídeo)")
    ap.add_argument("--output", default="output/court_keypoints.png")
    ap.add_argument("--draw-court", action="store_true",
                    help="dibuja el esqueleto de pista uniendo keypoints")
    ap.add_argument("--device", default=None, help="cuda|cpu (sobrescribe config)")
    ap.add_argument("-v", "--verbose", action="store_true", help="logging detallado")
    args = ap.parse_args(argv)

    setup_logging(verbose=args.verbose)
    if not args.video and not args.image:
        ap.error("indica --video o --image")

    cfg = load_config(args.config)
    ccfg = cfg["models"]["court"]
    if args.device:
        ccfg["device"] = args.device

    frame = grab_frame(args)

    detector = build_court_detector(ccfg)
    court = detector.detect(frame, frame_idx=args.frame)
    kps = court.keypoints
    n_ok = int(np.isfinite(kps).all(axis=1).sum())
    print(f"Keypoints detectados: {n_ok}/{len(kps)}  (válido={court.valid})")

    # Homografía a metros de la pista completa (chequeo de coherencia geométrica)
    cm = CourtModel(
        length=cfg["court_model"]["length"],
        singles_width=cfg["court_model"].get("width", 8.23),
        doubles_width=cfg["court_model"].get("doubles_width", 10.97),
    )
    court = cm.estimate_homography(court)
    print("Homografía a metros:", "OK" if court.homography is not None else "no calculada")

    out = frame.copy()
    if args.draw_court:
        out = draw_court_skeleton(out, kps)
    out = draw_keypoints(out, kps)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f"Imagen guardada en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
