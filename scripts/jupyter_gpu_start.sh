#!/bin/bash
# scripts/jupyter_gpu_start.sh

# ==============================================================================
# DESCRIPTION:
#   Launches a Jupyter Lab instance on a remote HPC compute node and outputs
#   the specific SSH tunneling command required to connect from a local machine.
#
# REQUIREMENTS:
#   - Must be run within an active SLURM allocation (salloc/sbatch).
#   - Python environment must exist at the defined JUPYTER_EXEC path.
# ==============================================================================

# --- Configuration ---
# <<< SHARED PATH!
JUPYTER_EXEC="/lustre/proyectos/p037/env_bdb/bin/jupyter-lab"
JUPYTER_PORT="9999"

# --- Script Logic ---

# 1. Check if the EXECUTABLE exists
if [ ! -f "$JUPYTER_EXEC" ]; then
    echo "Error: Jupyter executable not found at $JUPYTER_EXEC" >&2
    echo "Make sure you have run 'pip install jupyterlab' in the environment." >&2
    exit 1
fi

echo "--- Starting Jupyter Lab on port $JUPYTER_PORT ---"
NODE_NAME=$(hostname -f)

echo ""
echo "===================== INSTRUCTIONS ====================="
echo "1. Keep this terminal open."
echo "2. In ANOTHER terminal on your LOCAL PC (WSL), run the SSH tunnel:"
echo ""
echo "   ssh -N -L $JUPYTER_PORT:$NODE_NAME:$JUPYTER_PORT username@yuca.unison.mx"
echo ""
echo "3. Wait for Jupyter to start below."
echo "4. Copy one of the URLs containing 'http://127.0.0.1:$JUPYTER_PORT/lab?token=...'"
echo "5. Paste that URL into your local browser or VSCode."
echo "========================================================"
echo ""

# <<< Call 'jupyter-lab' using the full path
$JUPYTER_EXEC --no-browser --port="$JUPYTER_PORT" --ip=0.0.0.0
