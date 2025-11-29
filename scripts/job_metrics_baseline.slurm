#!/bin/bash
#SBATCH --job-name=metrics_baseline
#SBATCH --output=/lustre/home/dante/BigDataBowl/slurm_outputs/metrics_%j.out
#SBATCH --error=/lustre/home/dante/BigDataBowl/slurm_outputs/metrics_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G

echo "--- Starting Phase 4.2 (Baseline Metrics) ---"
date

###############
# 1. Activate environment
###############
export PATH="/lustre/proyectos/p037/env_bdb/bin:$PATH"
export LD_LIBRARY_PATH="/lustre/proyectos/p037/env_bdb/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/lustre/home/dante/BigDataBowl/src:$PYTHONPATH"

echo "Python being used:"
which python
python --version

###############
# 2. Run Phase 4.2 Script
###############
cd /lustre/home/dante/BigDataBowl/src
python compute_metrics_playlevel.py   

echo "--- Phase 4.2 Completed ---"
date
