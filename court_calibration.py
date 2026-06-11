#!/usr/bin/env python3
"""
Calibracion y deteccion de pista de padel para camara fija (RPi v3 wide).
Flujo:
  1. Cargas un frame (la camara es fija -> calibras UNA vez por sesion).
  2. Detectas las lineas blancas con vision clasica (HSV + FastLineDetector).
  3. Marcas a mano los vertices de pista para calcular la homografia imagen->pista (metros).
  4. Reutilizas esa homografia para proyectar pies de jugadores y botes de pelota
     a coordenadas reales de pista, de donde salen TODAS las estadisticas.

Sin TrackNet, sin GPU. Corre en la propia Raspberry Pi 5.

Uso:
  python court_calibration.py --frame frame.jpg --calibrate    # marcar puntos a mano
  python court_calibration.py --frame frame.jpg --lines        # solo ver deteccion de lineas
  python court_calibration.py --frame frame.jpg --project      # usar homografia guardada

Requisitos:
  pip install opencv-contrib-python numpy   (contrib trae ximgproc.FastLineDetector)
"""

import argparse
import json
import os
import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Coordenadas reales de una pista de padel (metros). Origen en una esquina.
# Pista: 10 m (ancho, eje X) x 20 m (largo, eje Y). Red en Y=10.
# Lineas de servicio a 3 m de cada fondo (Y=3 y Y=17). Linea central X=5
# entre las lineas de servicio. Ajusta segun lo que vea TU camara.
# ----------------------------------------------------------------------------
COURT_W = 10.0   # ancho en metros
COURT_L = 20.0   # largo en metros

# Puntos de referencia conocidos en la pista (metros). Marca estos mismos
# puntos, EN ESTE ORDEN, sobre la imagen durante la calibracion.
COURT_REFERENCE_POINTS = {
    "esquina_fondo_izq":   (0.0,  0.0),
    "esquina_fondo_der":   (COURT_W, 0.0),
    "servicio_izq":        (0.0,  3.0),
    "servicio_der":        (COURT_W, 3.0),
    "red_izq":             (0.0,  10.0),
    "red_der":             (COURT_W, 10.0),
}

HOMOGRAPHY_FILE = "homography.json"


# ----------------------------------------------------------------------------
# 1. Deteccion de lineas blancas (vision clasica, sin red neuronal)
# ----------------------------------------------------------------------------
def detect_white_lines(frame, min_len=60):
    """Devuelve segmentos de lineas blancas de la pista.

    Estrategia: aislar el blanco en HSV (alta luminosidad, baja saturacion),
    limpiar, y extraer segmentos rectos. La interseccion de segmentos largos
    da vertices con precision sub-pixel, mejor que keypoints de una CNN.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Blanco: S bajo, V alto. Ajusta estos rangos a tu iluminacion.
    lower = np.array([0, 0, 170])
    upper = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Limpieza morfologica para quedarse con lineas finas continuas.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    segments = []
    try:
        fld = cv2.ximgproc.createFastLineDetector(length_threshold=int(min_len))
        lines = fld.detect(mask)
        if lines is not None:
            segments = [l[0] for l in lines]
    except AttributeError:
        # Fallback si no tienes opencv-contrib: HoughLinesP
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=80,
                                minLineLength=min_len, maxLineGap=20)
        if lines is not None:
            segments = [l[0] for l in lines]

    return mask, segments


def draw_lines(frame, segments):
    out = frame.copy()
    for x1, y1, x2, y2 in segments:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return out


# ----------------------------------------------------------------------------
# 2. Calibracion manual: clicar los puntos de referencia sobre la imagen
# ----------------------------------------------------------------------------
def manual_calibration(frame):
    """Clica, en orden, los puntos de COURT_REFERENCE_POINTS. ENTER para confirmar,
    'u' para deshacer el ultimo, ESC para cancelar."""
    names = list(COURT_REFERENCE_POINTS.keys())
    clicked = []

    disp = frame.copy()
    win = "Calibracion - clica: " + names[0]

    def redraw():
        d = frame.copy()
        for i, (px, py) in enumerate(clicked):
            cv2.circle(d, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(d, names[i], (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(clicked) < len(names):
            cv2.setWindowTitle(win, "Clica: " + names[len(clicked)])
        cv2.imshow(win, d)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < len(names):
            clicked.append((x, y))
            redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        if k in (ord('u'), ord('U')) and clicked:
            clicked.pop()
            redraw()
        if k in (13, 10) and len(clicked) == len(names):  # ENTER
            break
    cv2.destroyAllWindows()

    img_pts = np.array(clicked, dtype=np.float32)
    court_pts = np.array([COURT_REFERENCE_POINTS[n] for n in names],
                         dtype=np.float32)
    return img_pts, court_pts


# ----------------------------------------------------------------------------
# 3. Homografia imagen <-> pista
# ----------------------------------------------------------------------------
def compute_homography(img_pts, court_pts):
    """H proyecta pixeles -> metros de pista. Usa RANSAC por robustez."""
    H, _ = cv2.findHomography(img_pts, court_pts, cv2.RANSAC, 5.0)
    return H


def image_to_court(H, pt):
    """(x_px, y_px) -> (x_m, y_m) en coordenadas de pista."""
    p = np.array([[pt]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def save_homography(H, img_pts, court_pts, path=HOMOGRAPHY_FILE):
    data = {
        "H": H.tolist(),
        "img_pts": img_pts.tolist(),
        "court_pts": court_pts.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[ok] Homografia guardada en {path}")


def load_homography(path=HOMOGRAPHY_FILE):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["H"], dtype=np.float32)


# ----------------------------------------------------------------------------
# 4. Helpers de estadisticas
# ----------------------------------------------------------------------------
def player_feet_from_mask(mask):
    """Pie del jugador = punto medio del borde inferior de la mascara de SAM3."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y_max = ys.max()
    x_at = int(np.mean(xs[ys == y_max]))
    return (x_at, int(y_max))


def draw_court_overlay(frame, H):
    """Dibuja una rejilla de la pista reproyectada a la imagen, para validar
    visualmente que la homografia es correcta."""
    Hinv = np.linalg.inv(H)
    out = frame.copy()
    # lineas en X cada 1 m, en Y cada 1 m
    for x in np.arange(0, COURT_W + 0.01, 1.0):
        pts = np.array([[[x, 0.0]], [[x, COURT_L]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2).astype(int)
        cv2.line(out, tuple(proj[0]), tuple(proj[1]), (255, 200, 0), 1)
    for y in np.arange(0, COURT_L + 0.01, 1.0):
        pts = np.array([[[0.0, y]], [[COURT_W, y]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2).astype(int)
        cv2.line(out, tuple(proj[0]), tuple(proj[1]), (255, 200, 0), 1)
    # red en Y=10
    pts = np.array([[[0.0, 10.0]], [[COURT_W, 10.0]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2).astype(int)
    cv2.line(out, tuple(proj[0]), tuple(proj[1]), (0, 0, 255), 2)
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="Imagen de un frame de tu video")
    ap.add_argument("--calibrate", action="store_true", help="Marcar puntos a mano")
    ap.add_argument("--lines", action="store_true", help="Ver deteccion de lineas")
    ap.add_argument("--project", action="store_true", help="Validar homografia guardada")
    args = ap.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        raise SystemExit(f"No pude abrir {args.frame}")

    if args.lines:
        mask, segments = detect_white_lines(frame)
        print(f"[info] {len(segments)} segmentos detectados")
        cv2.imshow("mascara blanco", mask)
        cv2.imshow("lineas", draw_lines(frame, segments))
        cv2.waitKey(0); cv2.destroyAllWindows()

    if args.calibrate:
        res = manual_calibration(frame)
        if res is None:
            raise SystemExit("Calibracion cancelada")
        img_pts, court_pts = res
        H = compute_homography(img_pts, court_pts)
        save_homography(H, img_pts, court_pts)
        # comprobacion: reproyectar puntos clicados a metros
        for (px, py), name in zip(img_pts, COURT_REFERENCE_POINTS):
            xm, ym = image_to_court(H, (px, py))
            print(f"  {name:20s} -> ({xm:5.2f} m, {ym:5.2f} m)")
        cv2.imshow("overlay pista", draw_court_overlay(frame, H))
        cv2.waitKey(0); cv2.destroyAllWindows()

    if args.project:
        H = load_homography()
        cv2.imshow("overlay pista", draw_court_overlay(frame, H))
        print("Ejemplo: centro de imagen ->",
              image_to_court(H, (frame.shape[1] / 2, frame.shape[0] / 2)), "m")
        cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
