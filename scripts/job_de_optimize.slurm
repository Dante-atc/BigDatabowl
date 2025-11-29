#!/bin/bash
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
# Agregamos PYTHONPATH por seguridad
export PYTHONPATH="/lustre/home/dante/BigDataBowl/src:$PYTHONPATH"

echo "Python version:"
which python
python --version

# ---------------------------------------------------------
# 2. Go to project directory
# ---------------------------------------------------------
cd /lustre/home/dante/BigDataBowl/src

# ---------------------------------------------------------
# 3. Run DE optimization script
# ---------------------------------------------------------
python de_optimize.py

echo "=== DE Optimization Completed ==="
date