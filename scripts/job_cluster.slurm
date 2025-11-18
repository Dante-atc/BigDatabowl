#!/bin/bash
#SBATCH --job-name=ssl_cluster
#SBATCH --output=/lustre/home/dante/BigDataBowl/slurm_outputs/cluster_%j.out
#SBATCH --error=/lustre/home/dante/BigDataBowl/slurm_outputs/cluster_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=64G

echo "--- Iniciando cluster ensemble ---"
date

# 1. Cargar ROCm
module load rocm/6.2.3

# 2. Activar tu entorno (usamos python3.9 del env_bdb directamente)
export PATH="/lustre/proyectos/p037/env_bdb/bin:$PATH"
export LD_LIBRARY_PATH="/lustre/proyectos/p037/env_bdb/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/lustre/home/dante/BigDataBowl/src:$PYTHONPATH"

which python
python --version

# 3. Ir al proyecto
cd /lustre/home/dante/BigDataBowl/src

# 4. Ejecutar el script de clustering
python cluster_embeddings.py

echo "--- Cluster ensemble completado ---"
date
