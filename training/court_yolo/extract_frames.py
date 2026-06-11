#!/usr/bin/env python3
"""Extrae frames de los vídeos para etiquetar la pista.

Como la cámara es FIJA, con pocos frames por cámara/sesión basta (el court no se
mueve). Conviene incluir variedad de luz y con/sin jugadores.

Uso:
    python extract_frames.py --video data/cam_left.mp4 --out dataset/raw/left --every 60
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=60, help="1 de cada N frames")
    ap.add_argument("--max", type=int, default=40, help="máx. frames a extraer")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {args.video}")
    base = os.path.splitext(os.path.basename(args.video))[0]
    i = saved = 0
    while saved < args.max:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.every == 0:
            path = os.path.join(args.out, f"{base}_{i:06d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
        i += 1
    cap.release()
    print(f"Guardados {saved} frames en {args.out}")


if __name__ == "__main__":
    main()
