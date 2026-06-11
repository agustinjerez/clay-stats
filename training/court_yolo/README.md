# Detección de pista con YOLOv8-pose (pista completa)

Entrena un modelo YOLO-pose que detecta los **14 keypoints de la pista completa**
(los dos fondos + líneas de saque). Con esos puntos y sus coordenadas reales se
calcula la homografía imagen → metros, base para situar botes (pelota) y golpes
(jugadores) y sacar las estadísticas con SAM3.

> Alternativa al modo `fixed` (calibración de 1 imagen, `tools/label_court_once.py`),
> útil si quieres robustez entre sesiones. Con cámara fija, `fixed` suele bastar.

## Orden de los 14 keypoints (vista de lado: media IZQUIERDA → media DERECHA)

La red queda vertical en el centro de la imagen. Se etiqueta de izquierda a
derecha y, en cada línea vertical, de la banda de arriba a la de abajo.

```
  FONDO IZQ   saque izq      red      saque der   FONDO DER
     0 ─────────── 4 ········|········ 7 ─────────── 10   (banda sup dobles)
     1 ─────────── ·         |         · ─────────── 11   (banda sup indiv)
     ·             5 ········|········ 8             ·    (T de saque)
     2 ─────────── ·         |         · ─────────── 12   (banda inf indiv)
     3 ─────────── 6 ········|········ 9 ─────────── 13   (banda inf dobles)

 0-3  media IZQ, fondo:  top_doubles, top_singles, bottom_singles, bottom_doubles
 4-6  media IZQ, saque:  top_singles, T, bottom_singles
 7-9  media DER, saque:  top_singles, T, bottom_singles
 10-13 media DER, fondo: top_doubles, top_singles, bottom_singles, bottom_doubles
```

(coincide con `src/analysis/court.py: COURT_KEYPOINT_NAMES`)

## Flujo

```bash
cd training/court_yolo
pip install ultralytics opencv-contrib-python

# 1. Extraer frames de cada cámara (la cámara es fija: bastan pocos)
python extract_frames.py --video ../../data/cam_left.mp4  --out dataset/raw/left
python extract_frames.py --video ../../data/cam_right.mp4 --out dataset/raw/right

# 2. Etiquetar los 13 keypoints (click en orden). ~15-30 imágenes por cámara.
python label_court.py --images dataset/raw/left  --out dataset --split train
python label_court.py --images dataset/raw/right --out dataset --split train
python label_court.py --images dataset/raw/left  --out dataset --split val   # algunas a val

# 3. Entrenar
python train.py --model yolov8n-pose.pt --epochs 120 --imgsz 1280

# 4. Usar los pesos
cp runs/pose/court_pose/weights/best.pt ../../weights/court_pose.pt
# y en config.yaml -> models.court.weights: "weights/court_pose.pt"
```

## Nota práctica

Como la cámara es **fija**, la pista está siempre en el mismo sitio dentro de cada
vídeo: con pocas imágenes por cámara el modelo aprende bien. Si solo vas a usar
una instalación fija, incluso podrías saltarte YOLO y marcar los 13 puntos una vez
(calibración manual) — pero YOLO te da robustez ante cambios de sesión, luz y
ligeros reajustes del trípode.
