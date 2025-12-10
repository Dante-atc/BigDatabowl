#!/bin/bash

# ==============================================================================
# DESCRIPTION:
#   Submits a GPU job to calculate baseline metrics for the NFL Big Data Bowl.
#   It sets up the Python path and library dependencies before execution.
# ==============================================================================

# --- SLURM Job Configuration ---
#SBATCH --job-name=metrics_baseline
#SBATCH --output=/lustre/home/dante/BigDataBowl/slurm_outputs/metrics_%j.out
#SBATCH --error=/lustre/home/dante/BigDataBowl/slurm_outputs/metrics_%j.err

#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G

echo "--- Starting Phase 4.2 (Baseline Metrics) ---"
date

# --- 1. Environment Configuration ---

# Add the project environment bin to the system PATH (Standardizes python calls)
export PATH="/lustre/proyectos/p037/env_bdb/bin:$PATH"

# Link dynamic libraries
export LD_LIBRARY_PATH="/lustre/proyectos/p037/env_bdb/lib:$LD_LIBRARY_PATH"

# Ensure Python can find your local modules in 'src'
export PYTHONPATH="/lustre/home/dante/BigDataBowl/src:$PYTHONPATH"

# Diagnostic checks
echo "Active Python executable:"
which python
python --version

# --- 2. Execution ---

# Navigate to source directory
cd /lustre/home/dante/BigDataBowl/src

echo "Running compute_metrics_playlevel.py..."
python compute_metrics_playlevel.py

echo "--- Phase 4.2 Completed ---"
date
