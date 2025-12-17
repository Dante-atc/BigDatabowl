#!/bin/bash
# ==============================================================================
# SCRIPT: gpu_session_start.sh
# DESCRIPTION:
#   Requests an interactive shell session on a GPU compute node.
#   Configures MIOpen cache directories for AMD GPUs to prevent permission
#   errors during interactive usage.
# ==============================================================================

# --- SLURM Configuration ---
PARTITION="gpu"
GRES="gpu:MI210:2"        # Request 2 GPUs (Specifically AMD MI210 models)
TIME="10:00:00"           # 10 hours of session time
SHELL_TYPE="/bin/bash"
ACCOUNT="p037"

# --- MIOpen Configuration (Required for AMD GPUs) ---
# Prevents "Disk Locked" errors when multiple users share a node
CACHE_DIR="$HOME/.miopen_cache"
USER_DB_DIR="$HOME/.miopen_user_db"

if [ ! -d "$CACHE_DIR" ]; then
    mkdir -p "$CACHE_DIR"
fi
if [ ! -d "$USER_DB_DIR" ]; then
    mkdir -p "$USER_DB_DIR"
fi

# Point MIOpen to your personal home folder instead of the locked system folder
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_DIR"
export MIOPEN_USER_DB_PATH="$USER_DB_DIR"

# --- SLURM Job Request ---
echo
echo " Requesting interactive GPU session..."
echo "   Project:   ${ACCOUNT}"
echo "   Partition: ${PARTITION}"
echo "   GPU:       ${GRES}"
echo "   Time:      ${TIME}"

# Execute the request
# --pty: Allocates a pseudo-terminal
srun --account="$ACCOUNT" --partition="$PARTITION" --gres="$GRES" --time="$TIME" --pty "$SHELL_TYPE"

echo " Session ended."
