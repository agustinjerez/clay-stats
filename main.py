#!/usr/bin/env python3
"""
Punto de entrada del pipeline de inferencia de tenis (TFM).

Uso:
    python main.py --config config.yaml
    python main.py --config config.yaml --video data/otro_partido.mp4
    python main.py --config config.yaml --upload     # sube el JSON al servicio web

Nota: la detección de PISTA está desactivada (en desarrollo, YOLO-pose). El
pipeline detecta pelota + jugadores con SAM3 y guarda las trayectorias en
píxeles; las estadísticas métricas se activarán al integrar la pista.
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from src.pipeline import TennisPipeline
from src.utils.logging_utils import setup_logging


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def upload_report(report: dict, endpoint: str) -> None:
    """Sube el JSON de estadísticas al servicio web."""
    import urllib.request

    data = json.dumps(report).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        print(f"[upload] {resp.status} -> {endpoint}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inferencia y estadísticas de tenis")
    parser.add_argument("--config", default="config.yaml", help="ruta al YAML de config")
    parser.add_argument("--video", default=None, help="sobrescribe el vídeo de entrada")
    parser.add_argument("--output", default=None, help="sobrescribe la ruta del JSON")
    parser.add_argument("--upload", action="store_true", help="subir el JSON al endpoint")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="logging detallado (DEBUG: por frame, por golpe, etc.)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="solo avisos y errores")
    args = parser.parse_args(argv)

    log = setup_logging(verbose=args.verbose, quiet=args.quiet)
    log.info("Iniciando pipeline | config=%s", args.config)
    cfg = load_config(args.config)
    if args.video:
        cfg["video"]["input_path"] = args.video
        cfg["video"].pop("cameras", None)
    if args.output:
        cfg["output"]["json_path"] = args.output

    pipeline = TennisPipeline(cfg)
    report = pipeline.run()

    def print_metrics(rep):
        s = rep["summary"]
        print(f"Golpes totales : {s['total_shots']}")
        print(f"Botes          : {s['total_bounces']}")
        print(f"Rallies        : {s['total_rallies']}")
        print(f"Rally más largo: {s['longest_rally_shots']} golpes")
        for pid, pd in rep["players"].items():
            print(f"  Jugador {pid}: {pd['total_shots']} golpes, "
                  f"rally máx {pd['longest_rally_shots']}, {pd['errors']} errores")

    if "summary" in report:                       # informe métrico (un vídeo)
        print("\n===== ESTADÍSTICAS =====")
        print_metrics(report)
    elif "sources" in report:                     # métrico multi-fuente
        print("\n===== ESTADÍSTICAS POR FUENTE =====")
        for label, rep in report["sources"].items():
            print(f"-- {label} --"); print_metrics(rep)
    else:                                         # informe de detecciones (sin pista)
        print("\n===== DETECCIONES (pista en desarrollo) =====")
        for label, cam in report.get("cameras", {}).items():
            b, p = cam["ball"], cam["players"]
            print(f"  [{label}] {cam['n_frames']} frames | pelota "
                  f"{b['visible']}/{b['frames_total']} ({b['visible_pct']}%) | "
                  f"jugadores: {p['num_tracks']} tracks, {p['num_detections']} det.")
    arts = report["_artifacts"]
    print(f"\nJSON guardado en: {arts['json']}")
    for label, vid in arts.get("annotated_video", {}).items():
        print(f"Vídeo anotado [{label}]: {vid}")

    endpoint = cfg["output"].get("upload_endpoint")
    if args.upload and endpoint:
        upload_report(report, endpoint)
    elif args.upload:
        print("[upload] No hay 'upload_endpoint' en la config.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
