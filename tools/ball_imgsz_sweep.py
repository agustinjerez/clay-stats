#!/usr/bin/env python3
"""Evaluación comparada de detectores: YOLO vs SAM3 (pelota y jugadores).

Calcula métricas estándar de detección y seguimiento contra un *ground truth*
etiquetado a mano (formato YOLO: un .txt por frame, `clase cx cy w h` normalizado):

  - mAP@0.5 y mAP@0.5:0.95   (AP estilo COCO, area bajo la curva P-R)
  - Precision / Recall / F1   (emparejado óptimo por IoU, umbral configurable)
  - IoU medio de los TP
  - MOTA / MOTP               (CLEAR-MOT con similitud IoU)

Notas sobre tracking (MOTA/MOTP):
  El GT en formato YOLO NO tiene IDs de seguimiento, así que MOTA/MOTP se calculan
  en su forma CLEAR-MOT con IoU pero SIN "ID switches" ni IDF1 (que requieren IDs).
  - MOTP = IoU medio de los emparejados (localización).
  - MOTA = 1 - (FN + FP + IDSW) / GT ; aquí IDSW = 0 (no hay IDs en el GT).
  Si algún día etiquetas en formato MOT (gt.txt con IDs), el cálculo se amplía
  solo (ver --gt-mot, pendiente de tu GT con IDs).

La PELOTA es casi un punto: además de la IoU (con una caja-proxy del tamaño
mediano del GT centrada en la detección) se reportan métricas por DISTANCIA de
centro (precision/recall/F1 @ umbral en px) y el error medio de posición (px),
que es la forma habitual de evaluar tracking de pelota (p. ej. TrackNet).

Uso típico:
    python tools/ball_imgsz_sweep.py --config config.yaml \
        --video data/clip.mp4 --labels-dir data/gt/labels \
        --backends yolo,sam3 --yolo-weights weights/yolov8x.pt

Verificación de la matemática (sin modelos, solo numpy/scipy):
    python tools/ball_imgsz_sweep.py --selftest
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Permite ejecutar el script desde cualquier sitio (añade la raíz del proyecto).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scipy.optimize import linear_sum_assignment
except Exception:                                  # pragma: no cover
    linear_sum_assignment = None


# ======================================================================
#  Geometría
# ======================================================================
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU de dos cajas [x1,y1,x2,y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Matriz IoU (n_gt x n_pred)."""
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)))
    M = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            M[i, j] = iou_xyxy(g, p)
    return M


def match_optimal(sim: np.ndarray, thr: float) -> List[Tuple[int, int]]:
    """Emparejamiento óptimo (Hungarian) maximizando similitud >= thr.

    sim: matriz (n_gt x n_pred). Devuelve lista de pares (i_gt, j_pred)."""
    if sim.size == 0:
        return []
    if linear_sum_assignment is None:                # fallback voraz
        pairs, used = [], set()
        order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
        rows, cols = set(), set()
        for i, j in order:
            if sim[i, j] < thr:
                break
            if i in rows or j in cols:
                continue
            pairs.append((int(i), int(j)))
            rows.add(i); cols.add(j)
        return pairs
    r, c = linear_sum_assignment(-sim)
    return [(int(i), int(j)) for i, j in zip(r, c) if sim[i, j] >= thr]


# ======================================================================
#  Estructuras de detección por frame
# ======================================================================
@dataclass
class FrameDet:
    boxes: np.ndarray                  # (n,4) xyxy
    scores: np.ndarray                 # (n,)


def _empty_frame() -> FrameDet:
    return FrameDet(np.zeros((0, 4)), np.zeros((0,)))


# ======================================================================
#  Métricas de detección (cajas)  ->  mAP, P/R, IoU, MOTA/MOTP
# ======================================================================
def average_precision(matched: List[bool], scores: List[float], n_gt: int) -> float:
    """AP = área bajo la curva precision-recall (interpolación a todos los puntos,
    estilo COCO). `matched[k]` = el k-ésimo pred (ordenado por score) es TP."""
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(matched, dtype=float)[order]
    fp = 1.0 - tp
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # envolvente monótona de la precisión
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_at_iou(gt_by_frame: Dict[int, np.ndarray],
              pred_by_frame: Dict[int, FrameDet], iou_thr: float) -> float:
    """AP a un umbral de IoU dado (matching voraz por score, cada GT una vez)."""
    flat = []   # (score, frame, j)
    for f, fd in pred_by_frame.items():
        for j in range(len(fd.boxes)):
            flat.append((float(fd.scores[j]), f, j))
    flat.sort(key=lambda x: -x[0])
    gt_used = {f: np.zeros(len(g), dtype=bool) for f, g in gt_by_frame.items()}
    matched, scores = [], []
    n_gt = sum(len(g) for g in gt_by_frame.values())
    for score, f, j in flat:
        scores.append(score)
        gts = gt_by_frame.get(f, np.zeros((0, 4)))
        pbox = pred_by_frame[f].boxes[j]
        best_i, best_iou = -1, iou_thr
        for i in range(len(gts)):
            if gt_used[f][i]:
                continue
            v = iou_xyxy(gts[i], pbox)
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            gt_used[f][best_i] = True
            matched.append(True)
        else:
            matched.append(False)
    return average_precision(matched, scores, n_gt)


def clearmot_and_pr(gt_by_frame: Dict[int, np.ndarray],
                    pred_by_frame: Dict[int, FrameDet],
                    iou_thr: float) -> dict:
    """Empareja por frame (óptimo) y agrega TP/FP/FN, IoU, MOTA, MOTP."""
    TP = FP = FN = 0
    iou_sum = 0.0
    frames = set(gt_by_frame) | set(pred_by_frame)
    for f in frames:
        g = gt_by_frame.get(f, np.zeros((0, 4)))
        fd = pred_by_frame.get(f, _empty_frame())
        sim = iou_matrix(g, fd.boxes)
        pairs = match_optimal(sim, iou_thr)
        TP += len(pairs)
        FN += len(g) - len(pairs)
        FP += len(fd.boxes) - len(pairs)
        iou_sum += sum(sim[i, j] for i, j in pairs)
    n_gt = sum(len(g) for g in gt_by_frame.values())
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    motp = iou_sum / max(TP, 1)                       # IoU medio de emparejados
    mota = 1.0 - (FN + FP + 0) / max(n_gt, 1)         # IDSW=0 (GT sin IDs)
    return dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1,
                mean_iou=motp, MOTP=motp, MOTA=mota, n_gt=n_gt)


def eval_boxes(gt_by_frame, pred_by_frame, iou_thr=0.5) -> dict:
    """Conjunto completo de métricas de detección para objetos con CAJA."""
    res = clearmot_and_pr(gt_by_frame, pred_by_frame, iou_thr)
    res["mAP50"] = ap_at_iou(gt_by_frame, pred_by_frame, 0.5)
    res["mAP5095"] = float(np.nanmean(
        [ap_at_iou(gt_by_frame, pred_by_frame, t) for t in np.arange(0.5, 1.0, 0.05)]))
    return res


# ======================================================================
#  Métricas de PELOTA (punto): distancia de centro + IoU proxy
# ======================================================================
def centers(fd: FrameDet) -> np.ndarray:
    if len(fd.boxes) == 0:
        return np.zeros((0, 2))
    return np.stack([(fd.boxes[:, 0] + fd.boxes[:, 2]) / 2,
                     (fd.boxes[:, 1] + fd.boxes[:, 3]) / 2], axis=1)


def eval_points(gt_by_frame: Dict[int, np.ndarray],
                pred_by_frame: Dict[int, FrameDet],
                dist_px: float, proxy_side: float) -> dict:
    """Métricas de pelota por DISTANCIA de centro (match si dist <= dist_px) +
    IoU proxy (caja cuadrada de lado `proxy_side` centrada en cada punto)."""
    TP = FP = FN = 0
    err_sum = 0.0
    iou_sum = 0.0
    matched_flags, scores_flat, n_gt = [], [], 0
    # para AP por distancia: aplanar predicciones ordenadas por score
    flat = []
    for f, fd in pred_by_frame.items():
        c = centers(fd)
        for j in range(len(c)):
            flat.append((float(fd.scores[j]), f, j))
    flat.sort(key=lambda x: -x[0])

    gt_centers = {f: (np.stack([(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2], 1)
                      if len(g) else np.zeros((0, 2))) for f, g in gt_by_frame.items()}
    n_gt = sum(len(g) for g in gt_by_frame.values())
    gt_used = {f: np.zeros(len(c), dtype=bool) for f, c in gt_centers.items()}

    # AP por distancia (greedy por score)
    for score, f, j in flat:
        scores_flat.append(score)
        gc = gt_centers.get(f, np.zeros((0, 2)))
        pc = centers(pred_by_frame[f])[j]
        best_i, best_d = -1, dist_px
        for i in range(len(gc)):
            if gt_used[f][i]:
                continue
            d = float(np.hypot(*(gc[i] - pc)))
            if d <= best_d:
                best_i, best_d = i, d
        if best_i >= 0:
            gt_used[f][best_i] = True
            matched_flags.append(True)
        else:
            matched_flags.append(False)
    ap_center = average_precision(matched_flags, scores_flat, n_gt)

    # P/R/IoU/MOTP/MOTA por frame (óptimo por distancia)
    frames = set(gt_by_frame) | set(pred_by_frame)
    for f in frames:
        gc = gt_centers.get(f, np.zeros((0, 2)))
        fd = pred_by_frame.get(f, _empty_frame())
        pc = centers(fd)
        if len(gc) and len(pc):
            D = np.zeros((len(gc), len(pc)))
            for i in range(len(gc)):
                for j in range(len(pc)):
                    D[i, j] = np.hypot(*(gc[i] - pc[j]))
            sim = np.maximum(0.0, 1.0 - D / dist_px)        # similitud en [0,1]
            pairs = match_optimal(sim, 1e-6)
            pairs = [(i, j) for (i, j) in pairs if D[i, j] <= dist_px]
        else:
            pairs = []
        TP += len(pairs)
        FN += len(gc) - len(pairs)
        FP += len(pc) - len(pairs)
        for i, j in pairs:
            err_sum += float(np.hypot(*(gc[i] - pc[j])))
            # IoU proxy: cajas cuadradas de lado proxy_side centradas en los puntos
            half = proxy_side / 2.0
            gbox = np.array([gc[i, 0] - half, gc[i, 1] - half, gc[i, 0] + half, gc[i, 1] + half])
            pbox = np.array([pc[j, 0] - half, pc[j, 1] - half, pc[j, 0] + half, pc[j, 1] + half])
            iou_sum += iou_xyxy(gbox, pbox)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1,
                pos_err_px=err_sum / max(TP, 1), mean_iou=iou_sum / max(TP, 1),
                MOTP=iou_sum / max(TP, 1), MOTA=1.0 - (FN + FP) / max(n_gt, 1),
                mAP50=ap_center, mAP5095=float("nan"), n_gt=n_gt)


# ======================================================================
#  Carga del ground truth (YOLO por frame)
# ======================================================================
def _frame_index(name: str) -> Optional[int]:
    m = re.findall(r"(\d+)", os.path.basename(name))
    return int(m[-1]) if m else None


def load_gt_yolo(labels_dir: str, W: int, H: int,
                 cls_map: Dict[int, str]) -> Dict[str, Dict[int, np.ndarray]]:
    """Lee labels YOLO. Devuelve {obj: {frame: (n,4) xyxy}} para cada clase de
    interés en cls_map = {class_id: 'ball'|'player'}."""
    out: Dict[str, Dict[int, np.ndarray]] = {v: {} for v in cls_map.values()}
    files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    for fn in files:
        fi = _frame_index(fn)
        if fi is None:
            continue
        rows = {v: [] for v in cls_map.values()}
        for line in open(os.path.join(labels_dir, fn), encoding="utf-8"):
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(float(p[0]))
            if cid not in cls_map:
                continue
            cx, cy, w, h = (float(x) for x in p[1:5])
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            rows[cls_map[cid]].append([x1, y1, x2, y2])
        for obj, lst in rows.items():
            out[obj][fi] = np.array(lst, dtype=float) if lst else np.zeros((0, 4))
    return out


def gt_median_box_side(gt_frames: Dict[int, np.ndarray]) -> float:
    sides = []
    for g in gt_frames.values():
        for b in g:
            sides.append(max(b[2] - b[0], b[3] - b[1]))
    return float(np.median(sides)) if sides else 10.0


# ======================================================================
#  Ejecutar detectores (poblar predicciones por frame)
# ======================================================================
def run_sam3_ball(models_cfg: dict, video: str) -> Dict[int, FrameDet]:
    from src.models import build_ball_detector
    c = dict(models_cfg["ball"]); c["backend"] = "sam3"
    det = build_ball_detector(c)
    obs = det.detect_video(video)
    side = float(c.get("_proxy_side", 10.0))
    out = {}
    for o in obs:
        if o.visible and o.x is not None:
            half = side / 2
            out[o.frame] = FrameDet(np.array([[o.x - half, o.y - half, o.x + half, o.y + half]]),
                                    np.array([o.score or 1.0]))
        else:
            out[o.frame] = _empty_frame()
    return out


def run_sam3_players(models_cfg: dict, video: str) -> Dict[int, FrameDet]:
    from src.models import build_player_detector
    c = dict(models_cfg["player"]); c["backend"] = "sam3"
    det = build_player_detector(c)
    pf = det.detect_video(video)
    out = {}
    for f, lst in pf.items():
        if lst:
            out[f] = FrameDet(np.array([o.bbox for o in lst], dtype=float),
                              np.array([o.score for o in lst], dtype=float))
        else:
            out[f] = _empty_frame()
    return out


def run_yolo(weights: str, video: str, class_id: int, conf: float,
             imgsz: int, device: str) -> Dict[int, FrameDet]:
    """Detección YOLO (Ultralytics) por frame, filtrada a `class_id`."""
    from ultralytics import YOLO
    model = YOLO(weights)
    results = model.predict(source=video, stream=True, conf=conf, classes=[class_id],
                            imgsz=imgsz, device=device, verbose=False)
    out = {}
    for i, r in enumerate(results):
        if r.boxes is not None and len(r.boxes) > 0:
            out[i] = FrameDet(r.boxes.xyxy.cpu().numpy(),
                              r.boxes.conf.cpu().numpy())
        else:
            out[i] = _empty_frame()
    return out


# ======================================================================
#  Salida
# ======================================================================
METRIC_ORDER = [("mAP50", "mAP@.5"), ("mAP5095", "mAP@.5:.95"),
                ("precision", "Prec"), ("recall", "Recall"), ("f1", "F1"),
                ("mean_iou", "IoU"), ("MOTA", "MOTA"), ("MOTP", "MOTP"),
                ("pos_err_px", "ErrPx")]


def print_table(title: str, results: Dict[str, dict]):
    print(f"\n=== {title} ===")
    cols = [lbl for key, lbl in METRIC_ORDER
            if any(key in r for r in results.values())]
    keys = [key for key, lbl in METRIC_ORDER
            if any(key in r for r in results.values())]
    head = f"{'backend':>8} | " + " | ".join(f"{c:>9}" for c in cols)
    print(head); print("-" * len(head))
    for backend, r in results.items():
        cells = []
        for k in keys:
            v = r.get(k, float("nan"))
            cells.append("   —   " if v is None or (isinstance(v, float) and np.isnan(v))
                         else f"{v:>9.3f}")
        print(f"{backend:>8} | " + " | ".join(cells))
    print(f"(GT: {next(iter(results.values())).get('n_gt','?')} objetos | "
          f"TP/FP/FN por backend abajo)")
    for backend, r in results.items():
        print(f"  {backend:>8}: TP={r.get('TP')}  FP={r.get('FP')}  FN={r.get('FN')}")


# ======================================================================
#  Self-test (valida la matemática sin modelos)
# ======================================================================
def selftest() -> int:
    print("== SELFTEST métricas ==")
    # caso perfecto: pred == gt
    gt = {0: np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
          1: np.array([[5, 5, 15, 15]])}
    pred = {f: FrameDet(g.copy(), np.ones(len(g))) for f, g in gt.items()}
    r = eval_boxes(gt, pred, 0.5)
    assert abs(r["precision"] - 1) < 1e-6 and abs(r["recall"] - 1) < 1e-6
    assert abs(r["mAP50"] - 1) < 1e-6 and abs(r["MOTA"] - 1) < 1e-6
    assert abs(r["mean_iou"] - 1) < 1e-6
    print("  perfecto -> P=R=mAP=MOTA=IoU=1  OK")

    # un FP y un FN: pred del frame1 desplazado (IoU 0), sobra una caja en frame0
    pred2 = {0: FrameDet(np.array([[0, 0, 10, 10], [20, 20, 30, 30], [100, 100, 110, 110]]),
                         np.array([0.9, 0.8, 0.7])),
             1: FrameDet(np.array([[80, 80, 90, 90]]), np.array([0.5]))}
    r2 = eval_boxes(gt, pred2, 0.5)
    # GT=3, TP=2, FP=2 (caja extra frame0 + caja mala frame1), FN=1
    assert r2["TP"] == 2 and r2["FP"] == 2 and r2["FN"] == 1, r2
    assert abs(r2["precision"] - 2 / 4) < 1e-6 and abs(r2["recall"] - 2 / 3) < 1e-6
    assert abs(r2["MOTA"] - (1 - (1 + 2) / 3)) < 1e-6
    print(f"  imperfecto -> P={r2['precision']:.3f} R={r2['recall']:.3f} "
          f"MOTA={r2['MOTA']:.3f}  OK")

    # IoU parcial: solapamiento 0.5 conocido
    gt3 = {0: np.array([[0, 0, 10, 10]])}
    pred3 = {0: FrameDet(np.array([[5, 0, 15, 10]]), np.array([1.0]))}
    # inter=5*10=50, union=100+100-50=150 -> IoU=1/3
    r3 = eval_boxes(gt3, pred3, 0.3)
    assert abs(r3["mean_iou"] - 1 / 3) < 1e-6, r3["mean_iou"]
    print(f"  IoU parcial -> {r3['mean_iou']:.3f} (esperado .333)  OK")

    # pelota por distancia
    gtb = {0: np.array([[100, 100, 110, 110]]), 1: np.array([[200, 200, 210, 210]])}
    predb = {0: FrameDet(np.array([[103, 104, 113, 114]]), np.array([0.9])),  # centro a ~5px
             1: _empty_frame()}                                               # FN
    rb = eval_points(gtb, predb, dist_px=8, proxy_side=10)
    assert rb["TP"] == 1 and rb["FN"] == 1 and rb["FP"] == 0, rb
    assert rb["pos_err_px"] < 8
    print(f"  pelota -> P={rb['precision']:.3f} R={rb['recall']:.3f} "
          f"err={rb['pos_err_px']:.2f}px  OK")
    print("TODOS LOS TESTS OK")
    return 0


# ======================================================================
#  Main
# ======================================================================
def _video_dims(video: str) -> Tuple[int, int]:
    import cv2
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return W, H


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluación YOLO vs SAM3 (pelota/jugadores)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", default=None, help="vídeo a evaluar (sobre el que está el GT)")
    ap.add_argument("--labels-dir", default=None, help="carpeta de labels YOLO (.txt por frame)")
    ap.add_argument("--backends", default="yolo,sam3", help="lista: yolo,sam3")
    ap.add_argument("--objects", default="ball,player", help="lista: ball,player")
    ap.add_argument("--gt-ball-class", type=int, default=0, help="id de 'pelota' en el GT")
    ap.add_argument("--gt-player-class", type=int, default=1, help="id de 'jugador' en el GT")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="IoU para P/R/MOTA (jugadores)")
    ap.add_argument("--ball-dist-px", type=float, default=None,
                    help="umbral de distancia de centro para la pelota (def: 2x lado mediano GT)")
    ap.add_argument("--yolo-weights", default="weights/yolov8x.pt")
    ap.add_argument("--yolo-ball-class", type=int, default=32, help="clase del modelo YOLO (COCO: 32=sports ball)")
    ap.add_argument("--yolo-player-class", type=int, default=0, help="clase YOLO (COCO: 0=person)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="guarda resultados a JSON")
    ap.add_argument("--selftest", action="store_true", help="valida la matemática y sale")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    import yaml
    from src.utils.logging_utils import setup_logging
    from src.utils.device import resolve_device
    setup_logging(verbose=False)

    if not args.video or not args.labels_dir:
        raise SystemExit("Indica --video y --labels-dir (GT YOLO). O usa --selftest.")
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    models_cfg = cfg["models"]
    device = resolve_device(args.device)

    W, H = _video_dims(args.video)
    cls_map = {args.gt_ball_class: "ball", args.gt_player_class: "player"}
    gt = load_gt_yolo(args.labels_dir, W, H, cls_map)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    print(f"Vídeo {args.video} ({W}x{H}) | GT frames pelota={len(gt.get('ball', {}))} "
          f"jugador={len(gt.get('player', {}))} | backends={backends}")

    ball_side = gt_median_box_side(gt.get("ball", {})) if "ball" in objects else 10.0
    ball_dist = args.ball_dist_px or (2.0 * ball_side)

    all_results = {}
    timings = {}
    for obj in objects:
        results = {}
        gt_frames = gt.get(obj, {})
        # solo evaluamos los frames etiquetados de este objeto
        labeled = set(gt_frames.keys())
        for backend in backends:
            t0 = time.perf_counter()
            if obj == "ball" and backend == "sam3":
                mc = dict(models_cfg); mc["ball"] = dict(models_cfg["ball"], _proxy_side=ball_side)
                preds = run_sam3_ball(mc, args.video)
            elif obj == "player" and backend == "sam3":
                preds = run_sam3_players(models_cfg, args.video)
            elif backend == "yolo":
                cls = args.yolo_ball_class if obj == "ball" else args.yolo_player_class
                imgsz = models_cfg["ball"].get("imgsz", 640) if obj == "ball" else \
                    models_cfg["player"].get("imgsz", 640)
                preds = run_yolo(args.yolo_weights, args.video, cls,
                                 conf=0.05, imgsz=imgsz, device=device)
            else:
                print(f"  (combinación no soportada: {backend}/{obj})"); continue
            timings[(obj, backend)] = time.perf_counter() - t0
            preds = {f: d for f, d in preds.items() if f in labeled}   # alinear a GT
            if obj == "ball":
                results[backend] = eval_points(gt_frames, preds, ball_dist, ball_side)
            else:
                results[backend] = eval_boxes(gt_frames, preds, args.iou_thr)
        if results:
            title = ("PELOTA  (dist<=%.0fpx, IoU-proxy lado=%.0fpx)" % (ball_dist, ball_side)
                     if obj == "ball" else "JUGADORES  (IoU>=%.2f)" % args.iou_thr)
            print_table(title, results)
            all_results[obj] = results

    print("\nTiempos (s):", {f"{o}/{b}": round(t, 1) for (o, b), t in timings.items()})
    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({o: {b: {k: (None if isinstance(v, float) and np.isnan(v) else v)
                               for k, v in r.items()}
                           for b, r in res.items()} for o, res in all_results.items()},
                      fh, indent=2, ensure_ascii=False)
        print("Resultados guardados en", args.out)
    print("\nNota MOTA/MOTP: GT YOLO sin IDs -> sin ID-switches/IDF1. MOTP=IoU medio "
          "de emparejados; MOTA=1-(FN+FP)/GT. Para CLEAR-MOT completo, GT en formato MOT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
