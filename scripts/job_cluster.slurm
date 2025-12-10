#!/bin/bash

# ==============================================================================
# SCRIPT: job_cluster.slurm
# DESCRIPTION:
#   Submits a batch job to perform Ensemble Clustering on the SSL embeddings.
#   It requests significant resources (32 CPUs, 64GB RAM) to handle large 
#   matrix operations and 1 GPU for hardware-accelerated dimensionality 
#   reduction (UMAP) or metric calculation.
# ==============================================================================

# --- SLURM Job Configuration ---
#SBATCH --job-name=ssl_cluster
#SBATCH --output=/lustre/home/dante/BigDataBowl/slurm_outputs/cluster_%j.out
#SBATCH --error=/lustre/home/dante/BigDataBowl/slurm_outputs/cluster_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=64G

echo "--- Starting Cluster Ensemble ---"
date

# ---------------------------------------------------------
# 1. Load Hardware Modules
# ---------------------------------------------------------
# Load ROCm (Radeon Open Compute) for AMD GPU acceleration
module load rocm/6.2.3

# ---------------------------------------------------------
# 2. Configure Environment
# ---------------------------------------------------------
export PYTHON_BIN="/lustre/proyectos/p037/env_bdb/bin/python"

# Add local source folder to Python path
export PYTHONPATH="/lustre/home/dante/BigDataBowl/src:$PYTHONPATH"

# Link dynamic libraries for GPU support
export LD_LIBRARY_PATH="/lustre/proyectos/p037/env_bdb/lib:$LD_LIBRARY_PATH"

echo "Python executable in use:"
echo $PYTHON_BIN
$PYTHON_BIN --version

# ---------------------------------------------------------
# 3. Navigate to Project Directory
# ---------------------------------------------------------
cd /lustre/home/dante/BigDataBowl/src

# ---------------------------------------------------------
# 4. Execute Clustering Script
# ---------------------------------------------------------
echo "Running cluster_embeddings.py..."
$PYTHON_BIN cluster_embeddings.py

echo "--- Cluster Ensemble Completed ---"
date
