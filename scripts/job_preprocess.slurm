#!/bin/bash

# --- Configuración del Trabajo SLURM ---
#SBATCH --job-name=preprocess_nfl
#SBATCH --output=slurm_outputs/preprocess_%j.out  
#SBATCH --error=slurm_outputs/preprocess_%j.err

#SBATCH --account=p037                   # Tu proyecto
#SBATCH --partition=cpu              # Partición de CPUs (o la que te indiquen)
#SBATCH --nodes=1                        # 1 Servidor
#SBATCH --ntasks-per-node=8              # 8 cores de CPU
#SBATCH --mem=32G                        # 32 GB de RAM (Pandas usa mucha)
#SBATCH --time=02:00:00                  # 2 horas de tiempo límite

echo "--- Iniciando trabajo de preprocesamiento ---"

# --- Comandos a ejecutar ---

# 1. Activar el entorno Conda
echo "Activando entorno..."
source /lustre/home/dante/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/proyectos/p037/env_bdb

# 2. Ir a la carpeta del script
cd /lustre/home/dante/BigDataBowl

# 3. Ejecutar el script de Python
echo "Corriendo script de Python..."
python src/preprocess_data.py

echo "--- Trabajo terminado ---"