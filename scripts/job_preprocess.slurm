#!/bin/bash

# ==============================================================================
# DESCRIPTION:
#   Submits a batch job to the SLURM scheduler to run the data preprocessing 
#   pipeline. It allocates CPU resources and invokes the specific project 
#   Python environment.
# ==============================================================================

# --- SLURM Job Configuration ---
#SBATCH --job-name=preprocess_nfl
#SBATCH --output=slurm_outputs/preprocess_%j.out  
#SBATCH --error=slurm_outputs/preprocess_%j.err

#SBATCH --account=p037                    
#SBATCH --partition=cpu                   
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=8               
#SBATCH --mem=32G                         
#SBATCH --time=02:00:00                   

echo "--- Starting preprocessing job ---"

# --- Commands to execute ---

# 1. Navigate to the script directory
cd /lustre/home/dante/BigDataBowl

# 2. Run the Python script USING THE ENVIRONMENT PYTHON
#    (This is the ONLY line that executes Python)
echo "Running Python script with env_bdb environment..."
/lustre/proyectos/p037/env_bdb/bin/python src/preprocess_data.py

echo "--- Job finished ---"
