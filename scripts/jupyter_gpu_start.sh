#!/bin/bash
# scripts/jupyter_gpu_start.sh

# --- Configuration ---
VENV_PATH="/lustre/proyectos/p037/env_bdb/bin/activate"
JUPYTER_PORT="9999"

# --- Script Logic ---

if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Entorno virtual no encontrado en $VENV_PATH" >&2
    exit 1
fi

echo "--- Activando Entorno Virtual ---"
source "$VENV_PATH"
echo "Entorno activo: $VIRTUAL_ENV"
echo ""
echo "--- Iniciando Jupyter Lab en el puerto $JUPYTER_PORT ---"

NODE_NAME=$(hostname -f)

echo ""
echo "===================== INSTRUCCIONES ====================="
echo "1. Mantén esta terminal abierta."
echo "2. En OTRA terminal en tu PC LOCAL (WSL), ejecuta el túnel SSH:"
echo ""
echo "   ssh -N -L $JUPYTER_PORT:$NODE_NAME:$JUPYTER_PORT dante@yuca.unison.mx"
echo ""
echo "3. Espera a que Jupyter inicie abajo."
echo "4. Copia una de las URLs que contienen 'http://127.0.0.1:$JUPYTER_PORT/lab?token=...'"
echo "5. Pega esa URL en tu navegador local o en VSCode."
echo "========================================================="
echo ""

jupyter lab --no-browser --port="$JUPYTER_PORT" --ip=0.0.0.0