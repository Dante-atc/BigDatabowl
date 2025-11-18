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

# 1. ROCm
module load rocm/6.2.3

# 2. Activar entorno correctamente
export PYTHON_BIN="/lustre/proyectos/p037/env_bdb/bin/python"
export PYTHONPATH="/lustre/home/dante/BigDataBowl/src:$PYTHONPATH"
export LD_LIBRARY_PATH="/lustre/proyectos/p037/env_bdb/lib:$LD_LIBRARY_PATH"

echo "Python que se usará:"
echo $PYTHON_BIN
$PYTHON_BIN --version

# 3. Ir al proyecto
cd /lustre/home/dante/BigDataBowl/src

# 4. Ejecutar script
$PYTHON_BIN cluster_embeddings.py

echo "--- Cluster ensemble completado ---"
date
