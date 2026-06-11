#!/usr/bin/env bash
# Crea el entorno virtual del proyecto e instala las dependencias.
# Uso:  ./setup_env.sh
set -e

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo ">> Usando: $($PY --version)"

if [ ! -d .venv ]; then
  echo ">> Creando entorno virtual en .venv"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Actualizando pip"
pip install --upgrade pip

echo ">> Instalando dependencias (requirements.txt)"
pip install -r requirements.txt

echo
echo "Hecho. Activa el entorno con:"
echo "    source .venv/bin/activate"
echo
echo "Nota: SAM3 no está en requirements (no publicado). El backend de pelota"
echo "por defecto es TrackNet. Para SAM3:"
echo "    pip install git+https://github.com/facebookresearch/sam3.git"
