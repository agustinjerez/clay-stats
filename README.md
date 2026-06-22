# Inferencia y estadísticas de tenis (TFM)

Pipeline en Python para analizar un vídeo de tenis que muestra la **pista
completa** (los dos fondos + la red) y producir un JSON de estadísticas listo
para subir a un servicio web.

## Qué hace

1. **Pista** — 14 keypoints de la **pista completa** + homografía imagen→metros.
   Dos modos (`config.yaml → models.court.mode`):
   - **`fixed`** (recomendado, cámara fija): calibras **una sola imagen** con
     `tools/label_court_once.py` y se reutiliza para todo el vídeo.
   - **`yolo`**: YOLOv8-pose entrenado (ver `training/court_yolo/`).
   Se activa con `court.enabled: true`; si está desactivada, el pipeline genera
   un informe de detecciones en píxeles (sin métricas).
2. **Pelota** — backends intercambiables vía `config.yaml → models.ball.backend`:
   - **`sam3`** (por defecto): tracking NATIVO de vídeo de SAM3 vía
     `SAM3VideoSemanticPredictor` (Ultralytics). SAM3 detecta y SIGUE la pelota
     con memoria temporal e identidades estables a lo largo del vídeo, en vez de
     re-detectar frame a frame. Prompt de texto + filtros geométricos.
   - **`sam3_image`**: SAM3 frame a frame (`SAM3SemanticPredictor`) + tracker
     propio por movimiento. Portado del notebook `sam3_tennis_ball_tracker.py`.
   - **`tracknet`**: TrackNetV2, alternativa sin GPU/SAM3.

   SAM3 requiere `ultralytics>=8.3.237`, GPU recomendada y `weights/sam3.pt`.

   Con `models.sam3_combined: true` (por defecto) y ambos backends en SAM3, la
   pelota y los jugadores se detectan en **una sola pasada** de SAM3 (prompt
   `["small yellow tennis ball", "tennis player"]`, separados por clase),
   reduciendo el tiempo casi a la mitad.

   Tras la detección, la trayectoria de la pelota se **refina**
   (`models.ball.refine`): filtro de Hampel + rechazo por velocidad para quitar
   falsos positivos, interpolación de huecos cortos y suavizado Savitzky-Golay.
   Mejora notablemente la detección de botes y golpes.
3. **Jugadores** — por defecto SAM3 (`SAM3VideoSemanticPredictor`, prompt
   `"tennis player"`), que detecta bien al jugador del fondo y da IDs estables;
   alternativa YOLO (`models.player.backend: "yolo"`). Los golpes se atribuyen al
   jugador más cercano al punto de impacto (con respaldo por lado de pista).

A partir de las trayectorias calcula y exporta:

- **Nº total de golpes por jugador**
- **Mapa de botes de la pelota en pista** (coordenadas en metros + heatmap PNG)
- **Rally de golpes más largo por jugador**
- **Errores por jugador**

Extras: velocidad media/máx de pelota por jugador, distancia recorrida, zonas
de bote (cuadrícula 3×3) y % dentro/fuera.

## Estructura

```
tennis_project/
├── main.py                  # CLI
├── config.yaml              # toda la configuración
├── requirements.txt
├── src/
│   ├── pipeline.py          # orquesta toda la inferencia y el análisis
│   ├── datatypes.py         # dataclasses compartidas
│   ├── models/
│   │   ├── court_detector.py   # CNN keypoints de pista
│   │   ├── ball_detector.py    # SAM3
│   │   └── player_detector.py  # YOLO
│   ├── analysis/
│   │   ├── court.py            # homografía y modelo de pista
│   │   ├── bounce_detector.py  # botes
│   │   ├── shot_detector.py    # golpes + atribución a jugador
│   │   ├── rally.py            # rallies + errores
│   │   └── statistics.py       # estadísticas + export JSON/heatmap
│   └── utils/                  # vídeo y geometría
└── output/                     # JSON y heatmap generados
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# SAM3 (cuando esté publicado):
pip install git+https://github.com/facebookresearch/sam3.git
```

Coloca los pesos en `weights/`:

- `weights/model_tennis_court_det.pt` — pesos de la CNN de pista (BallTrackerNet
  del repo `tennis_court_detector_git`).
- `weights/tracknet_v2.pth` — pesos de TrackNetV2 (backend de pelota por defecto).
  Si el fichero no existe, el modelo arranca con pesos aleatorios para validar
  el pipeline de extremo a extremo (no detectará bien la pelota). Para resultados
  reales, entrena TrackNetV2 con tus clips o usa unos pesos preentrenados.
- `weights/sam3.pt` (+ `configs/sam3.yaml`) — checkpoint de SAM3 (backend alternativo).
- `weights/sam3.pt` también se usa para los jugadores (backend por defecto).
- `weights/yolov8x.pt` — YOLO de jugadores (solo si `player.backend: "yolo"`).

## Uso

```bash
python main.py --config config.yaml
python main.py --config config.yaml --left data/cam_left.mp4 --right data/cam_right.mp4
python main.py --config config.yaml --upload      # POST del JSON al endpoint
python main.py --config config.yaml -v            # verbose (DEBUG)
python main.py --config config.yaml -q            # silencioso (solo avisos/errores)
```

Logging: el proyecto imprime por terminal cada fase (carga de modelos, detección
de pelota/pista/jugadores, análisis y export) con tiempos. `-v/--verbose` añade
detalle por frame y por evento; `-q/--quiet` reduce a avisos y errores. La
configuración está centralizada en `src/utils/logging_utils.py`.

## Puntos a ajustar a tu modelo

- **Orden de keypoints**: `src/analysis/court.py → reference_keypoints()` ya está
  alineado al orden canónico del repo `tennis_court_detector_git` (14 puntos, en
  metros). La arquitectura y el postproceso están en `src/models/court_tracknet.py`.
- **Resolución del vídeo**: el postproceso escala los keypoints con
  `sx=W/640, sy=H/360`, así funciona con cualquier resolución (incluido el
  panorámico 2558×720), no solo 1280×720 como el repo original.
- **API de SAM3**: `src/models/ball_detector.py` asume una API estilo
  video-predictor (como SAM2). Adáptala a la API oficial al publicarse.
- **Heurísticas de golpe/bote/error**: parámetros en `config.yaml → analysis`.

## Formato del JSON de salida

```json
{
  "match_id": "match_0001",
  "summary": { "total_shots": 0, "longest_rally_shots": 0, ... },
  "players": {
    "1": { "total_shots": 0, "longest_rally_shots": 0, "errors": 0,
           "avg_shot_speed_kmh": null, "distance_covered_m": 0.0 }
  },
  "bounce_map": { "points": [ {"x_m": 0, "y_m": 0, "inside": true, "side": "near"} ],
                  "inside": 0, "outside": 0, "zones_3x3": {} },
  "rallies": [ {"num_shots": 0, "ended_by": "out", "error_player": 1} ]
}
```
