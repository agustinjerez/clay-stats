"""
Pega dos vídeos lado a lado, sin warp ni homografía. Solo recorta y concatena.

Picker interactivo con 3 sliders y vista en vivo del resultado:
  - cut LEFT  : columna donde cortar el vídeo izquierdo (se queda 0..cut_L)
  - cut RIGHT : columna donde cortar el vídeo derecho   (se queda cut_R..fin)
  - vshift R  : desplazamiento vertical del derecho en px (positivo = baja)

Mueve los sliders hasta que la línea roja de costura caiga donde quieras y
las líneas de la pista crucen "más o menos" continuas.

Uso:
  python tennis_concat.py izq.mp4 der.mp4 -o pista.mp4

Saltar el picker reutilizando valores ya conocidos:
  python tennis_concat.py izq.mp4 der.mp4 -o pista.mp4 \
      --cut-l 1820 --cut-r 240 --vshift 0
"""

"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def concat_frame(fl, fr, cut_l, cut_r, vshift):
    #Concatena horizontalmente fl[:, :cut_l] y fr[:, cut_r:] con vshift en R.
    L = fl[:, :cut_l]
    R = fr[:, cut_r:]

    # Desplazamiento vertical del derecho
    if vshift != 0:
        H, W = R.shape[:2]
        shifted = np.zeros_like(R)
        if vshift > 0:
            v = min(vshift, H)
            shifted[v:, :] = R[:H - v, :]
        else:
            v = min(-vshift, H)
            shifted[:H - v, :] = R[v:, :]
        R = shifted

    # Igualar alturas (recorte)
    h = min(L.shape[0], R.shape[0])
    L, R = L[:h], R[:h]
    if L.shape[1] == 0:  return R
    if R.shape[1] == 0:  return L
    return np.hstack([L, R])


def pick_cuts(fl, fr, max_w=1600):
    #Devuelve (cut_l, cut_r, vshift) en coords originales, o None si se cancela.
    h_l, w_l = fl.shape[:2]
    h_r, w_r = fr.shape[:2]

    win = "concat picker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Defaults: sin recorte, sin shift
    cv2.createTrackbar("cut LEFT",  win, w_l - 1, w_l - 1, lambda v: None)
    cv2.createTrackbar("cut RIGHT", win, 0,       w_r - 1, lambda v: None)
    # vshift va de -200 a +200 representado como 0..400 (offset 200)
    SHIFT_RANGE = 200
    cv2.createTrackbar("vshift R +200", win, SHIFT_RANGE, 2 * SHIFT_RANGE, lambda v: None)

    state = {"done": False, "cancel": False}

    while True:
        cut_l  = max(1, cv2.getTrackbarPos("cut LEFT",  win))
        cut_r  = min(w_r - 1, cv2.getTrackbarPos("cut RIGHT", win))
        vshift = cv2.getTrackbarPos("vshift R +200", win) - SHIFT_RANGE

        merged = concat_frame(fl, fr, cut_l, cut_r, vshift)
        # Línea roja de costura
        seam_x = cut_l
        cv2.line(merged, (seam_x, 0), (seam_x, merged.shape[0]), (0, 0, 255), 2)

        # Reescalar para pantalla
        s = min(1.0, max_w / merged.shape[1])
        disp = cv2.resize(merged, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else merged

        info = (f"cut_L={cut_l}  cut_R={cut_r}  vshift={vshift}  "
                f"out={merged.shape[1]}x{merged.shape[0]}   [s]ave  [q]uit")
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(disp, info, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)

        k = cv2.waitKey(30) & 0xFF
        if k == ord('s'):
            state["done"] = True
            break
        elif k in (ord('q'), 27):
            state["cancel"] = True
            break

    cv2.destroyAllWindows()
    if state["cancel"] or not state["done"]:
        return None
    return cut_l, cut_r, vshift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left",  help="Vídeo izquierdo")
    ap.add_argument("right", help="Vídeo derecho")
    ap.add_argument("-o", "--output", default="concat.mp4")
    ap.add_argument("--frame", type=int, default=None,
                    help="Frame para el picker (def: medio)")
    ap.add_argument("--cut-l",  type=int, help="Saltar picker: x de corte izquierdo")
    ap.add_argument("--cut-r",  type=int, help="Saltar picker: x de corte derecho")
    ap.add_argument("--vshift", type=int, default=0, help="Desplazamiento vertical de R en px")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--scale", type=float, default=1.0, help="Escala del vídeo de salida")
    args = ap.parse_args()

    cap_l = cv2.VideoCapture(args.left)
    cap_r = cv2.VideoCapture(args.right)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise SystemExit("No se pudo abrir alguno de los vídeos.")

    fps    = cap_l.get(cv2.CAP_PROP_FPS) or 25.0
    n_proc = min(int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT)),
                 int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.max_frames:
        n_proc = min(n_proc, args.max_frames)

    # Frame de referencia para el picker
    fidx = args.frame if args.frame is not None else n_proc // 2
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, fidx); ok_l, fl0 = cap_l.read()
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, fidx); ok_r, fr0 = cap_r.read()
    if not (ok_l and ok_r):
        raise SystemExit(f"No se pudo leer el frame {fidx}")

    if args.cut_l is not None and args.cut_r is not None:
        cut_l, cut_r, vshift = args.cut_l, args.cut_r, args.vshift
        print(f"[cuts] cut_L={cut_l}  cut_R={cut_r}  vshift={vshift}  (sin picker)")
    else:
        print("Picker: ajusta los sliders. 's' para guardar, 'q' para salir.")
        result = pick_cuts(fl0, fr0)
        if result is None:
            print("Cancelado.")
            return
        cut_l, cut_r, vshift = result
        print(f"[cuts] cut_L={cut_l}  cut_R={cut_r}  vshift={vshift}")

    # Dimensión de salida basada en un frame de muestra
    sample = concat_frame(fl0, fr0, cut_l, cut_r, vshift)
    out_h, out_w = sample.shape[:2]
    out_w_s, out_h_s = int(out_w * args.scale), int(out_h * args.scale)
    print(f"[out] {out_w_s} x {out_h_s} @ {fps:.2f} fps  ({n_proc} frames)")

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w_s, out_h_s))
    if not writer.isOpened():
        raise SystemExit("No se pudo crear el vídeo de salida.")

    cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_proc):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break
        merged = concat_frame(fl, fr, cut_l, cut_r, vshift)
        # Por si algún frame difiere mínimamente del esperado:
        if merged.shape[:2] != (out_h, out_w):
            merged = cv2.resize(merged, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if args.scale != 1.0:
            merged = cv2.resize(merged, (out_w_s, out_h_s), interpolation=cv2.INTER_AREA)
        writer.write(merged)
        if i % 50 == 0:
            print(f"  frame {i}/{n_proc}")

    cap_l.release(); cap_r.release(); writer.release()
    print(f"[ok] {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()"""


"""
Pega dos vídeos lado a lado, sin warp ni homografía. Solo recorta, sincroniza y concatena.

Picker interactivo con sliders y vista en vivo:
  - frame      : posición del vídeo (para buscar un evento de referencia)
  - offset     : desplazamiento de cam2 respecto a cam1, en frames
                 (offset > 0 -> cam2 va por delante -> se le saltan N frames al exportar)
  - cut LEFT   : columna donde cortar el vídeo izquierdo (se queda 0..cut_L)
  - cut RIGHT  : columna donde cortar el vídeo derecho   (se queda cut_R..fin)
  - vshift R   : desplazamiento vertical del derecho en px (positivo = baja)

Uso:
  python tennis_concat.py izq.mp4 der.mp4 -o pista.mp4

Saltar el picker reutilizando valores ya conocidos:
  python tennis_concat.py izq.mp4 der.mp4 -o pista.mp4 \
      --cut-l 1820 --cut-r 240 --vshift 0 --offset 3

Atajos en el picker:
  s            guardar y salir
  q  /  ESC    cancelar
  ← / →        +-1 frame
  , / .        +-1 offset
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def concat_frame(fl, fr, cut_l, cut_r, vshift):
    """Concatena horizontalmente fl[:, :cut_l] y fr[:, cut_r:] con vshift en R."""
    L = fl[:, :cut_l]
    R = fr[:, cut_r:]

    if vshift != 0:
        H = R.shape[0]
        shifted = np.zeros_like(R)
        if vshift > 0:
            v = min(vshift, H)
            shifted[v:, :] = R[:H - v, :]
        else:
            v = min(-vshift, H)
            shifted[:H - v, :] = R[v:, :]
        R = shifted

    h = min(L.shape[0], R.shape[0])
    L, R = L[:h], R[:h]
    if L.shape[1] == 0:  return R
    if R.shape[1] == 0:  return L
    return np.hstack([L, R])


def read_frame_at(cap, idx, n_max):
    """Lee el frame idx de cap, con clamp [0, n_max-1]. Devuelve (ok, frame, idx_clamped)."""
    idx = max(0, min(n_max - 1, idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return ok, frame, idx


def pick_params(cap_l, cap_r, n_max, max_w=1600):
    """Picker con frame, offset, cuts y vshift. Devuelve (cut_l, cut_r, vshift, offset) o None."""
    # Frame inicial para obtener dimensiones
    ok_l, fl, _ = read_frame_at(cap_l, n_max // 2, n_max)
    ok_r, fr, _ = read_frame_at(cap_r, n_max // 2, n_max)
    if not (ok_l and ok_r):
        return None

    h_l, w_l = fl.shape[:2]
    h_r, w_r = fr.shape[:2]

    win = "concat + sync picker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    OFFSET_RANGE = 120     # +/- 120 frames (~2 s @ 60 fps); súbelo si tus cámaras pueden ir más desfasadas
    SHIFT_RANGE  = 200

    cv2.createTrackbar("frame",        win, n_max // 2,    max(1, n_max - 1), lambda v: None)
    cv2.createTrackbar("offset +120",  win, OFFSET_RANGE,  2 * OFFSET_RANGE,  lambda v: None)
    cv2.createTrackbar("cut LEFT",     win, w_l - 1,       w_l - 1,           lambda v: None)
    cv2.createTrackbar("cut RIGHT",    win, 0,             w_r - 1,           lambda v: None)
    cv2.createTrackbar("vshift +200",  win, SHIFT_RANGE,   2 * SHIFT_RANGE,   lambda v: None)

    state = {"done": False, "cancel": False}
    last_fidx, last_offset = None, None

    while True:
        fidx   = cv2.getTrackbarPos("frame", win)
        offset = cv2.getTrackbarPos("offset +120", win) - OFFSET_RANGE
        cut_l  = max(1, cv2.getTrackbarPos("cut LEFT",  win))
        cut_r  = min(w_r - 1, cv2.getTrackbarPos("cut RIGHT", win))
        vshift = cv2.getTrackbarPos("vshift +200", win) - SHIFT_RANGE

        # Solo re-leemos frames si cambió la posición temporal
        if fidx != last_fidx or offset != last_offset:
            ok_l, new_fl, _      = read_frame_at(cap_l, fidx,          n_max)
            ok_r, new_fr, r_used = read_frame_at(cap_r, fidx + offset, n_max)
            if ok_l: fl = new_fl
            if ok_r: fr = new_fr
            last_fidx, last_offset = fidx, offset

        merged = concat_frame(fl, fr, cut_l, cut_r, vshift)
        cv2.line(merged, (cut_l, 0), (cut_l, merged.shape[0]), (0, 0, 255), 2)

        s = min(1.0, max_w / merged.shape[1])
        disp = cv2.resize(merged, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else merged

        info1 = (f"frame={fidx}  offset={offset:+d}  "
                 f"cut_L={cut_l}  cut_R={cut_r}  vshift={vshift}  "
                 f"out={merged.shape[1]}x{merged.shape[0]}")
        info2 = "[s]ave  [q]uit   arrows: +-1 frame   ,/. : +-1 offset"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 56), (0, 0, 0), -1)
        cv2.putText(disp, info1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(disp, info2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(win, disp)

        k = cv2.waitKey(30) & 0xFF
        if k == ord('s'):
            state["done"] = True; break
        elif k in (ord('q'), 27):
            state["cancel"] = True; break
        elif k in (81, ord('a')):                          # ← / a
            cv2.setTrackbarPos("frame", win, max(0, fidx - 1))
        elif k in (83, ord('d')):                          # → / d
            cv2.setTrackbarPos("frame", win, min(n_max - 1, fidx + 1))
        elif k == ord(','):
            cv2.setTrackbarPos("offset +120", win, max(0, (offset - 1) + OFFSET_RANGE))
        elif k == ord('.'):
            cv2.setTrackbarPos("offset +120", win, min(2 * OFFSET_RANGE, (offset + 1) + OFFSET_RANGE))

    cv2.destroyAllWindows()
    if state["cancel"] or not state["done"]:
        return None
    return cut_l, cut_r, vshift, offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left",  help="Vídeo izquierdo")
    ap.add_argument("right", help="Vídeo derecho")
    ap.add_argument("-o", "--output", default="concat.mp4")
    ap.add_argument("--cut-l",  type=int, help="Saltar picker: x de corte izquierdo")
    ap.add_argument("--cut-r",  type=int, help="Saltar picker: x de corte derecho")
    ap.add_argument("--vshift", type=int, help="Desplazamiento vertical de R en px")
    ap.add_argument("--offset", type=int,
                    help="Sync: frames que cam2 (R) va por delante de cam1 (L). "
                         "Positivo -> se saltan N frames de R; negativo -> de L.")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    cap_l = cv2.VideoCapture(args.left)
    cap_r = cv2.VideoCapture(args.right)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise SystemExit("No se pudo abrir alguno de los vídeos.")

    fps   = cap_l.get(cv2.CAP_PROP_FPS) or 25.0
    n_l   = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    n_r   = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    n_max = min(n_l, n_r)

    # ¿Saltar picker?
    all_given = all(v is not None for v in (args.cut_l, args.cut_r, args.vshift, args.offset))
    if all_given:
        cut_l, cut_r, vshift, offset = args.cut_l, args.cut_r, args.vshift, args.offset
        print(f"[params] cut_L={cut_l}  cut_R={cut_r}  vshift={vshift}  offset={offset:+d}  (sin picker)")
    else:
        print("Picker: ajusta los sliders. 's' para guardar, 'q' para cancelar.")
        res = pick_params(cap_l, cap_r, n_max)
        if res is None:
            print("Cancelado."); return
        cut_l, cut_r, vshift, offset = res
        print(f"[params] cut_L={cut_l}  cut_R={cut_r}  vshift={vshift}  offset={offset:+d}")

    # Aplicar sync: descartar frames iniciales del que va adelantado
    if offset >= 0:
        start_l, start_r = 0, offset
    else:
        start_l, start_r = -offset, 0
    n_proc = min(n_l - start_l, n_r - start_r)
    if args.max_frames:
        n_proc = min(n_proc, args.max_frames)
    print(f"[sync] start_L={start_l}  start_R={start_r}  procesando {n_proc} frames")

    # Frame de muestra ya sincronizado para dimensionar salida
    _, fl0, _ = read_frame_at(cap_l, start_l, n_l)
    _, fr0, _ = read_frame_at(cap_r, start_r, n_r)
    sample = concat_frame(fl0, fr0, cut_l, cut_r, vshift)
    out_h, out_w = sample.shape[:2]
    out_w_s, out_h_s = int(out_w * args.scale), int(out_h * args.scale)
    print(f"[out] {out_w_s} x {out_h_s} @ {fps:.2f} fps  ({n_proc} frames)")

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w_s, out_h_s))
    if not writer.isOpened():
        raise SystemExit("No se pudo crear el vídeo de salida.")

    # Posicionar ambos caps al primer frame sincronizado y leer secuencialmente
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, start_l)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, start_r)

    for i in range(n_proc):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break
        merged = concat_frame(fl, fr, cut_l, cut_r, vshift)
        if merged.shape[:2] != (out_h, out_w):
            merged = cv2.resize(merged, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if args.scale != 1.0:
            merged = cv2.resize(merged, (out_w_s, out_h_s), interpolation=cv2.INTER_AREA)
        writer.write(merged)
        if i % 50 == 0:
            print(f"  frame {i}/{n_proc}")

    cap_l.release(); cap_r.release(); writer.release()
    print(f"[ok] {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()