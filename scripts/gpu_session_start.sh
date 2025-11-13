#!/bin/bash
# scripts/gpu_session_start.sh

# --- SLURM Configuration ---
PARTITION="gpu"
GRES="gpu:MI210:2"        # Pide 2 GPUs
TIME="10:00:00"           # 10 horas de tiempo
SHELL_TYPE="/bin/bash"
ACCOUNT="p037"

# --- MIOpen Configuration (para GPUs AMD) ---
CACHE_DIR="$HOME/.miopen_cache"
USER_DB_DIR="$HOME/.miopen_user_db"

if [ ! -d "$CACHE_DIR" ]; then
    mkdir -p "$CACHE_DIR"
fi
if [ ! -d "$USER_DB_DIR" ]; then
    mkdir -p "$USER_DB_DIR"
fi
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_DIR"
export MIOPEN_USER_DB_PATH="$USER_DB_DIR"

# --- SLURM Job Request ---
echo
echo "🚀 Solicitando sesion interactiva con GPU..."
echo "   Proyecto: ${ACCOUNT}"
echo "   Particion: ${PARTITION}"
echo "   GPU: ${GRES}"
echo "   Tiempo: ${TIME}"

srun --account="$ACCOUNT" --partition="$PARTITION" --gres="$GRES" --time="$TIME" --pty "$SHELL_TYPE"

echo "👋 Sesion terminada."