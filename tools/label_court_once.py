#!/usr/bin/env python3
"""Calibración de PISTA COMPLETA de UNA sola imagen (cámara fija).

Como la cámara no se mueve, la pista está siempre en el mismo sitio: basta
etiquetar los 19 keypoints de la pista entera UNA vez. Se guardan en un JSON
(con la resolución de la imagen) que el pipeline reutiliza para todo el vídeo.

Vista de lado: red vertical en el centro, media pista IZQUIERDA y DERECHA.
Orden: 0-13 fondos y líneas de saque (izq->der, banda arriba->abajo); luego
14-18 = intersección de las 5 líneas longitudinales con la RED (arriba->abajo).

Controles:
    click izq    : coloca el keypoint actual (visible)
    click der    : BORRA el keypoint más cercano al cursor (para recolocarlo)
    n            : marca el keypoint actual como NO visible y avanza
    u            : deshacer el último colocado
    g            : guardar JSON y salir
    q            : salir sin guardar

Uso:
    python tools/label_court_once.py --image data/images/frame_00000.jpg \
        --out weights/court_keypoints.json
"""
import argparse
import json
import os

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="imagen de referencia (frame del vídeo)")
    ap.add_argument("--out", default="weights/court_keypoints.json")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"No se pudo leer {args.image}")
    h, w = img.shape[:2]
    DELETE_RADIUS = 25                       # px para borrar con click derecho
    pts = [None] * K                         # cada slot: (x, y, v) o None
    state = {"cur": 0}

    def next_empty():
        for i in range(K):
            if pts[i] is None:
                return i
        return K

    def redraw():
        disp = img.copy()
        for i, p in enumerate(pts):
            if p is None:
                continue
            x, y, v = p
            color = (0, 0, 255) if v > 0 else (120, 120, 120)
            if v > 0:
                cv2.circle(disp, (int(x), int(y)), 5, color, -1)
            cv2.putText(disp, str(i), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        done = sum(1 for p in pts if p is not None)
        cur = state["cur"]
        nxt = KEYPOINTS[cur] if cur < K else "COMPLETO (g=guardar)"
        msg = f"[{done}/{K}] siguiente: {nxt}   (click der = borrar)"
        cv2.rectangle(disp, (0, 0), (920, 30), (0, 0, 0), -1)
        cv2.putText(disp, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("court", disp)

    def nearest_idx(x, y):
        best, best_d = None, DELETE_RADIUS ** 2
        for i, p in enumerate(pts):
            if p is None or p[2] <= 0:
                continue
            d = (p[0] - x) ** 2 + (p[1] - y) ** 2
            if d < best_d:
                best, best_d = i, d
        return best

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["cur"] < K:
            pts[state["cur"]] = (float(x), float(y), 2)
            state["cur"] = next_empty()
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = nearest_idx(x, y)
            if i is not None:
                pts[i] = None
                state["cur"] = next_empty()   # recolocar desde el hueco creado
                redraw()

    cv2.namedWindow("court", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("court", on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n") and state["cur"] < K:
            pts[state["cur"]] = (0.0, 0.0, 0)
            state["cur"] = next_empty()
            redraw()
        elif key == ord("u"):
            filled = [i for i, p in enumerate(pts) if p is not None]
            if filled:
                pts[filled[-1]] = None
                state["cur"] = next_empty()
                redraw()
        elif key == ord("q"):
            print("Cancelado."); break
        elif key == ord("g"):
            kps = [[round(p[0], 2), round(p[1], 2)] if (p and p[2] > 0) else None
                   for p in pts]
            data = {"source_image": args.image, "width": w, "height": h,
                    "keypoint_names": KEYPOINTS, "keypoints": kps}
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(data, f, indent=2)
            n_ok = sum(1 for k in kps if k is not None)
            print(f"Guardado {args.out} ({n_ok}/{K} keypoints, {w}x{h})")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
