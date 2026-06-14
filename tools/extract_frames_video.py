""" # extract_frames.py
import cv2, os

video_path = 'input_videos/pano_parque_corto_720.mp4'
out_dir = 'tennis_court_detector_git/data/images_parque_corto'
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# extraer 1 frame cada 3 segundos -> ~variedad sin redundancia
#step = int(fps * 3)
step = int(fps) # temporal para probar el proceso con menos frames
saved = 0
for i in range(0, total, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok: continue
    # reescalar a 1280x720 si hace falta
    frame = cv2.resize(frame, (1280, 720))
    cv2.imwrite(f'{out_dir}/frame_{saved:05d}.png', frame)
    saved += 1
cap.release()
print(f'guardados {saved} frames') """

# extract_frames.py
import cv2, os

#video_path = 'input_videos/pano_parque_corto_720.mp4'
video_path = '/Users/agustin_jerez/MasterIA/TFM/tennis_project_claude/data/pano_last_1080_3.mp4'
out_dir = '/Users/agustin_jerez/MasterIA/TFM/tennis_project_claude/tools/imgz' #tennis_court_detector_git/data/images_parque_corto
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f'No se pudo abrir el vídeo: {video_path}')

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Vídeo: {w}x{h} @ {fps:.2f} fps, {total} frames')

# extraer 1 frame cada N segundos
seconds_between = 5   # cambia a 3 cuando quieras menos redundancia
step = max(1, int(round(fps * seconds_between)))

## cambiar con cada salto
saved = 0
for i in range(0, total, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        continue
    # solo redimensionar si hace falta
    if (w, h) != (1280, 720):
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{out_dir}/frame_{saved:05d}.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved += 1

cap.release()
print(f'guardados {saved} frames en {out_dir}')