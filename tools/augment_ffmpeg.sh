#!/usr/bin/env bash
# ============================================================
#  Data augmentation FOTOMÉTRICA con ffmpeg para entrenar CNNs
#  - Mantiene resolución y fps originales
#  - NO mueve píxeles -> las etiquetas (keypoints/pelota/jugadores)
#    siguen siendo válidas. (Para flips/crops habría que transformar labels.)
#
#  Compatible con bash 3.2 (macOS por defecto): sin arrays asociativos.
#
#  Uso:
#     ./augment_ffmpeg.sh entrada.mp4 [carpeta_salida]
# ============================================================
set -eo pipefail

IN="${1:?Pasa el vídeo de entrada: ./augment_ffmpeg.sh entrada.mp4 [salida]}"
OUT="${2:-augmented}"
mkdir -p "$OUT"

BASE="$(basename "${IN%.*}")"
ENC="-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an"

# Lista "nombre|filtro". Una variante por línea.
VARIANTS='
gray|hue=s=0
contrast_high|eq=contrast=1.4
contrast_low|eq=contrast=0.7
bright|eq=brightness=0.12
dark|eq=brightness=-0.12
saturated|eq=saturation=1.6
desaturated|eq=saturation=0.4
gamma_up|eq=gamma=1.4
gamma_down|eq=gamma=0.7
hue_warm|hue=h=25
hue_cool|hue=h=-25
cb_warm|colorbalance=rm=0.2:gm=0.05:bm=-0.2
cb_cool|colorbalance=rm=-0.2:gm=0.0:bm=0.2
noise|noise=alls=18:allf=t+u
blur|gblur=sigma=1.6
sharpen|unsharp=5:5:1.0:5:5:0.0
sepia|colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131
combo1|eq=brightness=0.06:contrast=1.2:saturation=1.3:gamma=0.9,noise=alls=10:allf=t
combo2|hue=h=15:s=1.2,eq=contrast=1.15,gblur=sigma=1.0
'

count=0
echo "$VARIANTS" | while IFS='|' read -r name filt; do
  [ -z "$name" ] && continue
  out="$OUT/${BASE}__${name}.mp4"
  echo ">> $name -> $out"
  ffmpeg -nostdin -y -v error -i "$IN" -vf "$filt" $ENC "$out"
  count=$((count + 1))
done

echo "Hecho. Variantes en: $OUT/"
