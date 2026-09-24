#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Egg Quality - Deploy remoto (ejecutar DENTRO de la VM, un solo comando)
# -----------------------------------------------------------------------------
# Supone que ya están en la VM:
#   /home/ubuntu/egg_quality_models.zip        (push con deploy_local.ps1 o scp)
#   /home/ubuntu/damaged_eggs_CNN/api/         (carpeta api del repo)
#
# Usa `uv` para crear un entorno con Python 3.12 (TensorFlow no tiene wheels
# para el Python 3.14 que trae la Ubuntu actual, por eso no usamos el python3
# del sistema).
#
# Uso:
#   bash deploy_remote.sh [/ruta/al/egg_quality_models.zip] [MODELO.keras]
# =============================================================================

ZIP="${1:-/home/ubuntu/egg_quality_models.zip}"
MODEL_FILE="${2:-mejor_mobilenet.keras}"
MODEL_VERSION="${MODEL_VERSION:-v1.0}"
PY_VERSION="${PY_VERSION:-3.12}"

API_DIR=/home/ubuntu/damaged_eggs_CNN/api
MODELS_DIR=/home/ubuntu/egg_quality_models
SERVICE=egg-quality

echo "==> 1/7 Dependencias del sistema"
sudo apt-get update -qq
sudo apt-get install -y -qq curl unzip

echo "==> 2/7 Descomprimir modelo"
if [[ ! -f "$ZIP" ]]; then
    echo "No encuentro $ZIP. Pásalo al script o usa scp antes." >&2
    exit 1
fi
rm -rf "$MODELS_DIR"
mkdir -p "$MODELS_DIR"
unzip -q -o -d "$MODELS_DIR" "$ZIP"
echo "Modelos disponibles:"
ls -1 "$MODELS_DIR"/*.keras || echo "(ningún .keras en el zip)"

echo "==> 3/7 API"
if [[ ! -d "$API_DIR" ]]; then
    mkdir -p "$API_DIR"
    echo "No encuentro $API_DIR. Copia la carpeta api/ del repo a la VM." >&2
    exit 1
fi

echo "==> 4/7 Python $PY_VERSION + dependencias (uv)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
rm -rf "$API_DIR/.venv"
uv venv --python "$PY_VERSION" "$API_DIR/.venv"
uv pip install --python "$API_DIR/.venv/bin/python" --upgrade pip
uv pip install --python "$API_DIR/.venv/bin/python" -r "$API_DIR/requirements.txt"

echo "==> 5/7 Servicio systemd ($SERVICE)"
sudo tee /etc/systemd/system/$SERVICE.service >/dev/null <<EOF
[Unit]
Description=Egg Quality API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=$API_DIR
Environment=MODEL_PATH=$MODELS_DIR/$MODEL_FILE
Environment=MODEL_VERSION=$MODEL_VERSION
ExecStart=$API_DIR/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now $SERVICE

echo "==> 6/7 Verificar servicio"
sleep 5
sudo systemctl reset-failed $SERVICE 2>/dev/null || true
sudo systemctl status $SERVICE --no-pager | tail -n 12 || true

echo "==> 7/7 Health check (local)"
curl -fsS http://localhost:8000/health

echo ""
echo "Listo. Desde el celular:"
echo "  http://54.227.194.211:8000/health"
echo "Modelo servido: $MODELS_DIR/$MODEL_FILE"