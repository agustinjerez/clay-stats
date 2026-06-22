#!/usr/bin/env python3
"""Cobertura de detección sobre TODO el vídeo (sin etiquetar nada a mano).

Recorre todos los frames y cuenta:
  - en cuántos se detecta la PELOTA,
  - en cuántos se detectan 2 (o más) JUGADORES.

No necesita ground truth: solo cuenta lo que detecta el modelo. Funciona con
SAM3 o con YOLO (Ultralytics).

Filtro de PISTA (por defecto activado): descarta detecciones fuera del polígono
de la pista (envolvente de los keypoints calibrados, weights/court_keypoints.json)
+ un margen. Así no cuenta al público ni a personas del fondo como jugadores, ni
falsos positivos de pelota en vallas/fondo. Desactívalo con --no-roi.

Uso:
    python tools/ball_imgsz_sweep.py                       # SAM3, vídeo de config
    python tools/ball_imgsz_sweep.py --backend yolo --yolo-weights weights/yolov8x.pt
    python tools/ball_imgsz_sweep.py --backend both        # compara los dos
    python tools/ball_imgsz_sweep.py --no-roi              # sin filtro de pista
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class FrameData:
    ball: Optional[Tuple[float, float]] = None          # centro (x,y) o None
    players: List[Tuple[float, float, float]] = field(default_factory=list)  # foot_x, foot_y, score


# ======================================================================
#  Detección con SAM3
# ======================================================================
def detect_sam3(models_cfg: dict, video: str, imgsz: int = None,
                ball_prompt: str = None) -> List[FrameData]:
    from src.models import build_ball_detector, build_player_detector

    bcfg = dict(models_cfg["ball"]); bcfg["backend"] = "sam3"
    if imgsz:
        bcfg["imgsz"] = imgsz
    if ball_prompt:
        bcfg["prompt"] = ball_prompt
    ball_obs = build_ball_detector(bcfg).detect_video(video)
    n = len(ball_obs)
    frames = [FrameData() for _ in range(n)]
    for o in ball_obs:
        if o.visible and o.x is not None and 0 <= o.frame < n:
            frames[o.frame].ball = (float(o.x), float(o.y))

    pcfg = dict(models_cfg["player"]); pcfg["backend"] = "sam3"
    if imgsz:
        pcfg["imgsz"] = imgsz
    pf = build_player_detector(pcfg).detect_video(video)
    for fr, lst in pf.items():
        if 0 <= fr < n:
            frames[fr].players = [(float(p.foot_x), float(p.foot_y), float(p.score)) for p in lst]
    return frames


# ======================================================================
#  Detección con YOLO (una pasada: pelota + personas)
# ======================================================================
def detect_yolo(weights: str, video: str, ball_cls: int, person_cls: int,
                conf: float, imgsz: int, device: str,
                ball_name: str = "tennis ball", person_name: str = "person") -> List[FrameData]:
    from ultralytics import YOLO

    model = YOLO(weights)
    # Modelos open-vocab (YOLO-World/YOLOE): aceptan clases por TEXTO, así que se
    # puede pedir "tennis ball". Con un modelo COCO normal no existe esa clase y
    # se cae a la clase numérica (32 = 'sports ball').
    open_vocab = hasattr(model, "set_classes")
    if open_vocab:
        try:
            model.set_classes([ball_name, person_name])
            ball_cls, person_cls = 0, 1                 # índices según la lista dada
            print(f"  YOLO open-vocab -> clases por texto: ['{ball_name}', '{person_name}']")
        except Exception as e:                          # noqa: BLE001
            print(f"  (aviso: set_classes falló, uso clases COCO: {e})")
            open_vocab = False
    if not open_vocab:
        print(f"  YOLO COCO -> pelota=clase {ball_cls}, persona=clase {person_cls} "
              f"(para 'tennis ball' usa un modelo YOLO-World)")
    results = model.predict(source=video, stream=True, conf=conf,
                            classes=[ball_cls, person_cls], imgsz=imgsz,
                            device=device, verbose=False)
    frames: List[FrameData] = []
    for r in results:
        fd = FrameData()
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            balls = [(b, c) for b, c, k in zip(xyxy, cf, cls) if k == ball_cls]
            if balls:                                   # pelota: la de mayor confianza
                b, _ = max(balls, key=lambda z: z[1])
                fd.ball = (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2))
            for b, c, k in zip(xyxy, cf, cls):
                if k == person_cls:                     # pie = centro inferior de la caja
                    fd.players.append((float((b[0] + b[2]) / 2), float(b[3]), float(c)))
        frames.append(fd)
    return frames


# ======================================================================
#  Filtro de pista (ROI) — polígono de los keypoints calibrados
# ======================================================================
def build_court_roi(video: str, court_cfg: dict):
    """Devuelve (hull, bbox_h) del polígono de pista, o None si no se puede."""
    import cv2
    import numpy as np
    from src.models import build_court_detector

    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    try:
        det = build_court_detector(court_cfg)
        cf = det.detect(frame, 0)
    except Exception as e:                              # noqa: BLE001
        print(f"  (aviso: sin filtro de pista, no se pudo cargar la calibración: {e})")
        return None
    kp = cf.keypoints[np.isfinite(cf.keypoints).all(axis=1)].astype(np.float32)
    if len(kp) < 4:
        print("  (aviso: sin filtro de pista, calibración con <4 keypoints)")
        return None
    hull = cv2.convexHull(kp)
    bbox_h = float(kp[:, 1].max() - kp[:, 1].min())
    return hull, bbox_h


def apply_roi(frames: List[FrameData], roi, margin_player_frac: float,
              margin_ball_frac: float, do_ball: bool, do_player: bool) -> List[FrameData]:
    """Filtra por el polígono de pista. do_ball/do_player activan cada filtro.

    Por defecto solo jugadores: la pelota se cuenta en bruto (cuadra con el log
    del detector). El filtro de pelota descartaría lobs/saques fuera del polígono.
    """
    if roi is None or (not do_ball and not do_player):
        return frames
    import cv2
    hull, bbox_h = roi
    mp = margin_player_frac * bbox_h
    mb = margin_ball_frac * bbox_h
    out = []
    for fd in frames:
        g = FrameData()
        if fd.ball is not None:
            if do_ball:
                g.ball = fd.ball if cv2.pointPolygonTest(hull, fd.ball, True) >= -mb else None
            else:
                g.ball = fd.ball
        if do_player:
            g.players = [p for p in fd.players
                         if cv2.pointPolygonTest(hull, (p[0], p[1]), True) >= -mp]
        else:
            g.players = list(fd.players)
        out.append(g)
    return out


# ======================================================================
#  Cobertura + tabla comparativa por resolución
# ======================================================================
def coverage(frames: List[FrameData], target: int) -> dict:
    n = len(frames)
    return dict(
        n=n,
        ball=sum(1 for f in frames if f.ball is not None),
        p1=sum(1 for f in frames if len(f.players) >= 1),
        pge=sum(1 for f in frames if len(f.players) >= target),
        peq=sum(1 for f in frames if len(f.players) == target),
    )


def print_table(name: str, rows: List[Tuple[int, dict]], target: int,
                roi_desc: str, times: dict):
    """rows = [(imgsz, coverage_dict), ...]."""
    print(f"\n=== {name}  (filtro de pista -> {roi_desc}) ===")
    head = (f"{'imgsz':>6} | {'frames':>6} | {'pelota':>13} | {'>=1 jug':>13} | "
            f"{'>='+str(target)+' jug':>13} | {'=='+str(target)+' jug':>13} | {'tiempo s':>8}")
    print(head); print("-" * len(head))
    for imgsz, c in rows:
        n = c["n"]

        def pc(x):
            return f"{x:>5} ({100*x/max(n,1):>4.1f}%)"
        t = times.get(imgsz, float("nan"))
        print(f"{imgsz:>6} | {n:>6} | {pc(c['ball'])} | {pc(c['p1'])} | "
              f"{pc(c['pge'])} | {pc(c['peq'])} | {t:>8.1f}")


# ======================================================================
#  Main
# ======================================================================
def _snap_list(imgszs: List[int], stride: int) -> List[int]:
    """Redondea cada imgsz al múltiplo de `stride` más cercano (SAM3=14, YOLO=32),
    evita el aviso de Ultralytics y deja constancia del tamaño real usado."""
    out = []
    for v in imgszs:
        s = max(stride, int(round(v / stride)) * stride)
        if s != v:
            print(f"  imgsz {v} -> {s} (múltiplo de {stride})")
        if s not in out:
            out.append(s)
    return out



def main(argv=None):
    ap = argparse.ArgumentParser(description="Cobertura de detección (pelota / 2 jugadores)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", default=None,
                    help="vídeo a analizar (def: video.input_path de config.yaml)")
    ap.add_argument("--backend", default="sam3", choices=["sam3", "yolo", "both"])
    ap.add_argument("--imgszs", default="640,960,1280",
                    help="resoluciones de inferencia a comparar (separadas por comas)")
    ap.add_argument("--players", type=int, default=2, help="nº de jugadores objetivo")
    ap.add_argument("--no-roi", action="store_true",
                    help="desactiva el filtro de pista para los JUGADORES")
    ap.add_argument("--ball-roi", action="store_true",
                    help="aplica también el filtro de pista a la PELOTA "
                         "(por defecto no, para que cuadre con el log del detector)")
    ap.add_argument("--sam-ball-prompt", default="small yellow tennis ball",
                    help="prompt de texto de la pelota para SAM3")
    ap.add_argument("--yolo-weights", default="weights/yolov8x.pt")
    ap.add_argument("--yolo-ball-name", default="tennis ball",
                    help="clase de pelota por TEXTO (modelos YOLO-World/open-vocab)")
    ap.add_argument("--yolo-person-name", default="person")
    ap.add_argument("--yolo-ball-class", type=int, default=32,
                    help="clase numérica de pelota si el modelo es COCO (32=sports ball)")
    ap.add_argument("--yolo-person-class", type=int, default=0, help="COCO: 0=person")
    ap.add_argument("--conf", type=float, default=0.25, help="confianza mínima (YOLO)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    import yaml
    from src.utils.logging_utils import setup_logging
    from src.utils.device import resolve_device
    setup_logging(verbose=False)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    models_cfg = cfg["models"]
    video = args.video or cfg.get("video", {}).get("input_path")
    if not video:
        raise SystemExit("No hay vídeo: indica --video o pon video.input_path en config.yaml.")
    if not os.path.exists(video):
        raise SystemExit(f"No existe el vídeo: {video}")
    device = resolve_device(args.device)
    player_roi = not args.no_roi          # filtro de pista a jugadores (def. sí)
    ball_roi = args.ball_roi              # filtro de pista a la pelota (def. no)

    # Polígono de pista (una vez) para el filtro de zona.
    roi = None
    if player_roi or ball_roi:
        roi = build_court_roi(video, models_cfg.get("court", {}))
        if roi is None:
            player_roi = ball_roi = False
    mp_frac = models_cfg.get("player", {}).get("roi", {}).get("margin_frac", 0.4)
    mb_frac = models_cfg.get("ball", {}).get("roi", {}).get("margin_frac", 0.05)
    roi_desc = f"jugadores: {'sí' if player_roi else 'no'}, pelota: {'sí' if ball_roi else 'no'}"
    print(f"Vídeo: {video} | backend: {args.backend} | {roi_desc}")
    imgszs = [int(s) for s in args.imgszs.split(",") if s.strip()]

    if args.backend in ("sam3", "both"):
        rows, times = [], {}
        for imgsz in _snap_list(imgszs, 14):       # SAM3: múltiplos de 14
            t0 = time.perf_counter()
            frames = detect_sam3(models_cfg, video, imgsz, args.sam_ball_prompt)
            times[imgsz] = time.perf_counter() - t0
            frames = apply_roi(frames, roi, mp_frac, mb_frac, ball_roi, player_roi)
            rows.append((imgsz, coverage(frames, args.players)))
        print_table("SAM3", rows, args.players, roi_desc, times)
    if args.backend in ("yolo", "both"):
        rows, times = [], {}
        for imgsz in _snap_list(imgszs, 32):       # YOLO: múltiplos de 32
            t0 = time.perf_counter()
            frames = detect_yolo(args.yolo_weights, video, args.yolo_ball_class,
                                 args.yolo_person_class, args.conf, imgsz, device,
                                 args.yolo_ball_name, args.yolo_person_name)
            times[imgsz] = time.perf_counter() - t0
            frames = apply_roi(frames, roi, mp_frac, mb_frac, ball_roi, player_roi)
            rows.append((imgsz, coverage(frames, args.players)))
        print_table("YOLO", rows, args.players, roi_desc, times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
