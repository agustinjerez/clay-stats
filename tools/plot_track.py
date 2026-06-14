#!/usr/bin/env python3
"""Grafica la trayectoria de la pelota (X e Y en imagen) con los golpes y botes
detectados, a partir del CSV que vuelca el pipeline (output.dump_track: true).

Sirve para ver SIN ejecutar de nuevo: si la pelota tiene muchos huecos, si la X
revierte limpio (golpes) y si la Y tiene picos claros (botes), y si los umbrales
están bien.

Uso:
    python tools/plot_track.py --csv output/ball_track_video.csv
    python tools/plot_track.py --csv output/ball_track_video.csv --out output/track.png
"""
import argparse
import csv


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fr, x, y, vis, interp, bounce, shot = [], [], [], [], [], [], []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            fr.append(int(r["frame"]))
            x.append(float(r["x"]) if r["x"] else None)
            y.append(float(r["y"]) if r["y"] else None)
            vis.append(int(r["visible"]))
            interp.append(int(r["interpolated"]))
            bounce.append(int(r["is_bounce"]))
            shot.append(int(r["is_shot"]))

    n = len(fr)
    n_vis = sum(vis)
    n_real = sum(1 for v, i in zip(vis, interp) if v and not i)
    print(f"{n} frames | visible {n_vis} ({100*n_vis/max(1,n):.1f}%) | "
          f"detecciones reales {n_real} | interpoladas {n_vis-n_real} | "
          f"golpes {sum(shot)} | botes {sum(bounce)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fx = [f for f, xi in zip(fr, x) if xi is not None]
    xx = [xi for xi in x if xi is not None]
    fy = [f for f, yi in zip(fr, y) if yi is not None]
    yy = [yi for yi in y if yi is not None]

    ax1.plot(fx, xx, "-", color="tab:blue", lw=1)
    ax1.plot(fx, xx, ".", color="tab:blue", ms=3)
    for f, s in zip(fr, shot):
        if s:
            ax1.axvline(f, color="orange", lw=1.2, alpha=0.8)
    ax1.set_ylabel("X imagen (px)")
    ax1.set_title("Trayectoria horizontal (X). Líneas naranja = golpes detectados")

    ax2.plot(fy, yy, "-", color="tab:green", lw=1)
    ax2.plot(fy, yy, ".", color="tab:green", ms=3)
    for f, b in zip(fr, bounce):
        if b:
            ax2.axvline(f, color="red", lw=1.2, alpha=0.8)
    ax2.set_ylabel("Y imagen (px)")
    ax2.set_xlabel("frame")
    ax2.invert_yaxis()   # Y crece hacia abajo
    ax2.set_title("Trayectoria vertical (Y). Líneas rojas = botes detectados")

    fig.tight_layout()
    out = args.out or args.csv.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=110)
    print("Gráfico guardado en:", out)


if __name__ == "__main__":
    main()
