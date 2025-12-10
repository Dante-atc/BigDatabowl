#!/bin/bash

# ==============================================================================
# DESCRIPTION:
#   Submits a GPU job to extract SSL (Self-Supervised Learning) embeddings.
#   Loads the required ROCm module for AMD GPU acceleration and uses the
#   project-specific Python environment.
# ==============================================================================

# --- SLURM Job Configuration ---
#SBATCH --job-name=ssl_extract
#SBATCH --output=/lustre/home/dante/BigDataBowl/slurm_outputs/extract_%j.out
#SBATCH --error=/lustre/home/dante/BigDataBowl/slurm_outputs/extract_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

echo "--- Starting Embedding Extraction ---"
date

# --- 1. Environment & Modules ---

# Load specific software modules for the cluster hardware
# (ROCm is the open software platform for AMD GPU computing)
module load rocm/6.2.3

# Define the absolute path to the environment's Python executable
PYTHON_EXEC=/lustre/proyectos/p037/env_bdb/bin/python3.9

# --- 2. Execution ---

# Navigate to source directory
cd /lustre/home/dante/BigDataBowl/src

echo "Running extract_embeddings.py..."
$PYTHON_EXEC extract_embeddings.py

echo "--- Extraction Completed ---"
date
