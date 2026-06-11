#!/usr/bin/env python3
"""Entrena YOLOv8-pose para los keypoints de media pista.

Uso:
    python train.py                       # valores por defecto
    python train.py --model yolov8s-pose.pt --epochs 150 --imgsz 1280
"""
import argparse


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-pose.pt", help="modelo base pose")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1280)   # pista grande -> imgsz alto
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (auto si null)")
    ap.add_argument("--name", default="court_pose")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, name=args.name,
        # La pista es rígida: evita augmentations geométricas fuertes que rompan
        # la coherencia de los keypoints. Flip sí (usa flip_idx de data.yaml).
        degrees=0.0, shear=0.0, perspective=0.0, mosaic=0.0,
        fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )
    print("Entrenamiento terminado. Pesos en runs/pose/%s/weights/best.pt" % args.name)
    print("Cópialos a weights/ y apunta config.yaml -> models.court.weights")


if __name__ == "__main__":
    main()
