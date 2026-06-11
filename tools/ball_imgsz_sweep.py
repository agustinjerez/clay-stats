#!/usr/bin/env python3
"""Barrido de imgsz para la detección de pelota (SAM3).

Compara, sobre un CLIP corto del vídeo, el % de frames con pelota detectada y la
velocidad de proceso para varios tamaños de inferencia (imgsz). Así eliges el
mejor compromiso resolución/velocidad antes de lanzar el vídeo completo.

Uso:
    python tools/ball_imgsz_sweep.py --config config.yaml --seconds 6 \
        --imgszs 640,960,1280
    python tools/ball_imgsz_sweep.py --video data/match.mp4 --seconds 5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import yaml

# Permite ejecutar el script desde cualquier sitio (añade la raíz del proyecto).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_ball_detector
from src.utils.logging_utils import setup_logging


def trim_clip(src: str, seconds: float) -> str:
    """Recorta los primeros `seconds` a un fichero temporal con ffmpeg."""
    out = tempfile.mktemp(suffix=".mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-t", str(seconds), "-i", src,
           "-c:v", "libx264", "-an", out]
    subprocess.run(cmd, check=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Barrido de imgsz de la pelota")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", default=None, help="sobrescribe el vídeo de entrada")
    ap.add_argument("--imgszs", default="640,960,1280", help="lista separada por comas")
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="segundos del clip de prueba (0 = vídeo completo)")
    args = ap.parse_args(argv)

    setup_logging(verbose=False)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    vid = args.video or cfg["video"].get("input_path")
    if not vid:
        raise SystemExit("Indica --video o configura video.input_path")
    if not os.path.exists(vid):
        raise SystemExit(f"No existe el vídeo: {vid}")

    clip = trim_clip(vid, args.seconds) if args.seconds and args.seconds > 0 else vid
    imgszs = [int(s) for s in args.imgszs.split(",")]
    ball_cfg = dict(cfg["models"]["ball"])
    ball_cfg.setdefault("backend", "sam3")

    print(f"\nClip: {clip}  | imgszs: {imgszs}\n")
    print(f"{'imgsz':>6} | {'vis %':>6} | {'frames':>7} | {'tiempo s':>9} | {'fps':>6}")
    print("-" * 48)
    rows = []
    for imgsz in imgszs:
        c = dict(ball_cfg); c["imgsz"] = imgsz
        det = build_ball_detector(c)
        t0 = time.perf_counter()
        obs = det.detect_video(clip)
        dt = time.perf_counter() - t0
        n = len(obs)
        vis = sum(1 for o in obs if o.visible)
        pct = 100 * vis / max(1, n)
        fps = n / dt if dt > 0 else 0
        rows.append((imgsz, pct, n, dt, fps))
        print(f"{imgsz:>6} | {pct:>5.1f}% | {n:>7} | {dt:>9.1f} | {fps:>6.1f}")

    if args.seconds and clip != vid:
        os.remove(clip)
    best = max(rows, key=lambda r: r[1])
    print(f"\nMejor detección: imgsz={best[0]} ({best[1]:.1f}% pelota visible).")
    print("Sube imgsz mientras el % mejore de forma apreciable; si se estanca, "
          "no compensa el coste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
