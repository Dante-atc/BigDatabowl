#!/bin/bash

# ==============================================================================
# SCRIPT: job_de_optimize.slurm
# DESCRIPTION:
#   Submits a long-running job for Differential Evolution (DE) Optimization.
#   Allocates high CPU resources (16 cores) for parallel population evaluation.
# ==============================================================================

# --- SLURM Job Configuration ---
#SBATCH --job-name=de_optimize
#SBATCH --output=/lustre/home/dante/BigDataBowl/slurm_outputs/de_opt_%j.out
#SBATCH --error=/lustre/home/dante/BigDataBowl/slurm_outputs/de_opt_%j.err

#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --mem=32G

echo "=== Starting Differential Evolution Optimization (CPU Phase) ==="
date

# ---------------------------------------------------------
# 1. Environment Setup
# ---------------------------------------------------------

export PATH="/lustre/proyectos/p037/env_bdb/bin:$PATH"
export LD_LIBRARY_PATH="/lustre/proyectos/p037/env_bdb/lib:$LD_LIBRARY_PATH"

# Add PYTHONPATH to ensure local modules in 'src' are importable
export PYTHONPATH="/lustre/home/dante/BigDataBowl/src:$PYTHONPATH"

echo "Python version in use:"
which python
python --version

# ---------------------------------------------------------
# 2. Navigate to Project Directory
# ---------------------------------------------------------
cd /lustre/home/dante/BigDataBowl/src

# ---------------------------------------------------------
# 3. Run DE Optimization Script
# ---------------------------------------------------------
python de_optimize.py

echo "=== DE Optimization Completed ==="
date
