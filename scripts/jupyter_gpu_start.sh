#!/bin/bash
# scripts/jupyter_gpu_start.sh

# --- Configuration ---
# <<< ¡LA RUTA COMPARTIDA CORRECTA!
JUPYTER_EXEC="/lustre/proyectos/p037/env_bdb/bin/jupyter-lab"
JUPYTER_PORT="9999"

# --- Script Logic ---

# 1. Comprobamos que el EJECUTABLE exista
if [ ! -f "$JUPYTER_EXEC" ]; then
    echo "Error: El ejecutable de Jupyter no se encontró en $JUPYTER_EXEC" >&2
    echo "Asegúrate de haber corrido 'pip install jupyterlab' en el entorno." >&2
    exit 1
fi

echo "--- Iniciando Jupyter Lab en el puerto $JUPYTER_PORT ---"
NODE_NAME=$(hostname -f)

echo ""
echo "===================== INSTRUCCIONES ====================="
echo "1. Mantén esta terminal abierta."
echo "2. En OTRA terminal en tu PC LOCAL (WSL), ejecuta el túnel SSH:"
echo ""
echo "   ssh -N -L $JUPYTER_PORT:$NODE_NAME:$JUPYTER_PORT usuario@yuca.unison.mx"
echo ""
echo "3. Espera a que Jupyter inicie abajo."
echo "4. Copia una de las URLs que contienen 'http://127.0.0.1:$JUPYTER_PORT/lab?token=...'"
echo "5. Pega esa URL en tu navegador local o en VSCode."
echo "========================================================="
echo ""

# <<< Llamamos a 'jupyter-lab' usando la ruta completa
$JUPYTER_EXEC --no-browser --port="$JUPYTER_PORT" --ip=0.0.0.0