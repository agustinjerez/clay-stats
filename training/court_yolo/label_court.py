#!/usr/bin/env python3
"""Etiquetador de keypoints de MEDIA pista para YOLOv8-pose.

Para cada imagen, haz clic en los 13 keypoints EN ORDEN. La barra de estado
indica cuál toca. Genera etiquetas en formato YOLO-pose:

    0 cx cy w h  x1 y1 v1  x2 y2 v2 ... x13 y13 v13      (todo normalizado 0-1)

donde v = 2 (visible) o 0 (no visible / ocluido).

Teclas:
    click izq : coloca el keypoint actual (visible, v=2)
    n         : marca el keypoint actual como NO visible (v=0) y avanza
    u         : deshacer último
    g         : guardar y siguiente imagen
    s         : saltar imagen (no guardar)
    q         : salir

Uso:
    python label_court.py --images dataset/raw/left --out dataset --split train
"""
import argparse
import glob
import os
import shutil

import cv2

KEYPOINTS = [
    "left_baseline_top_doubles", "left_baseline_top_singles",
    "left_baseline_bottom_singles", "left_baseline_bottom_doubles",
    "left_service_top_singles", "left_service_T", "left_service_bottom_singles",
    "right_service_top_singles", "right_service_T", "right_service_bottom_singles",
    "right_baseline_top_doubles", "right_baseline_top_singles",
    "right_baseline_bottom_singles", "right_baseline_bottom_doubles",
    "net_top_doubles", "net_top_singles", "net_center",
    "net_bottom_singles", "net_bottom_doubles",
]
K = len(KEYPOINTS)


def label_image(img):
    pts = []   # (x, y, v)
    state = {"cur": 0}

    def redraw():
        disp = img.copy()
        for i, (x, y, v) in enumerate(pts):
            if v > 0:
                cv2.circle(disp, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.putText(disp, str(i), (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        msg = (f"[{len(pts)}/{K}] siguiente: "
               f"{KEYPOINTS[state['cur']] if state['cur'] < K else 'COMPLETO'}")
        cv2.rectangle(disp, (0, 0), (820, 30), (0, 0, 0), -1)
        cv2.putText(disp, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.imshow("label", disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["cur"] < K:
            pts.append((float(x), float(y), 2))
            state["cur"] += 1
            redraw()

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n") and state["cur"] < K:
            pts.append((0.0, 0.0, 0)); state["cur"] += 1; redraw()
        elif key == ord("u") and pts:
            pts.pop(); state["cur"] -= 1; redraw()
        elif key == ord("g"):
            return pts if len(pts) == K else None
        elif key == ord("s"):
            return "skip"
        elif key == ord("q"):
            return "quit"


def to_label(pts, w, h):
    xs = [p[0] for p in pts if p[2] > 0]
    ys = [p[1] for p in pts if p[2] > 0]
    if not xs:
        return None
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    pad = 0.02
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = min(1.0, (x2 - x1) / w + 2 * pad)
    bh = min(1.0, (y2 - y1) / h + 2 * pad)
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for (x, y, v) in pts:
        parts.append(f"{x / w:.6f} {y / h:.6f} {v}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="carpeta con frames a etiquetar")
    ap.add_argument("--out", default="dataset", help="raíz del dataset YOLO")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--sample", type=int, default=0,
                    help="etiquetar solo N frames repartidos uniformemente (0 = todos)")
    ap.add_argument("--start", type=int, default=0, help="saltar los primeros N frames")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images", args.split)
    lbl_dir = os.path.join(args.out, "labels", args.split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.png")))
    files = files[args.start:]
    if args.sample and len(files) > args.sample:
        step = len(files) / args.sample
        files = [files[int(i * step)] for i in range(args.sample)]
    print(f"{len(files)} imágenes a etiquetar. Teclas: click=punto, n=no-visible, "
          f"u=undo, g=guardar, s=saltar, q=salir")
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        res = label_image(img)
        if res == "quit":
            break
        if res == "skip" or res is None:
            continue
        label = to_label(res, w, h)
        if label is None:
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        shutil.copy(f, os.path.join(img_dir, os.path.basename(f)))
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as fh:
            fh.write(label + "\n")
        print("guardado", name)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
