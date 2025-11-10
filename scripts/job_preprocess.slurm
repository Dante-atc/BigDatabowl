#!/bin/bash

# --- Configuración del Trabajo SLURM ---
#SBATCH --job-name=preprocess_nfl
#SBATCH --output=slurm_outputs/preprocess_%j.out  
#SBATCH --error=slurm_outputs/preprocess_%j.err

#SBATCH --account=p037                   
#SBATCH --partition=cpu                  
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=8              
#SBATCH --mem=32G                        
#SBATCH --time=02:00:00                  

echo "--- Iniciando trabajo de preprocesamiento ---"

# --- Comandos a ejecutar ---

# 1. Ir a la carpeta del script
cd /lustre/home/dante/BigDataBowl

# 2. Ejecutar el script de Python USANDO EL PYTHON DEL ENTORNO
#    (Esta es la ÚNICA línea que ejecuta Python)
echo "Corriendo script de Python con el entorno env_bdb..."
/lustre/proyectos/p037/env_bdb/bin/python src/preprocess_data.py

echo "--- Trabajo terminado ---"