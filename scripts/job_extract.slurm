#!/bin/bash
#SBATCH --job-name=ssl_extract
#SBATCH --output=/lustre/home/dante/BigDataBowl/slurm_outputs/extract_%j.out
#SBATCH --error=/lustre/home/dante/BigDataBowl/slurm_outputs/extract_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

echo "--- Iniciando extracción de embeddings ---"
date

module load rocm/6.2.3

PYTHON_EXEC=/lustre/proyectos/p037/env_bdb/bin/python3.9

cd /lustre/home/dante/BigDataBowl/src

$PYTHON_EXEC extract_embeddings.py

echo "--- Extracción completada ---"
date
