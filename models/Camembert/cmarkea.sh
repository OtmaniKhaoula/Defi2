#!/bin/bash
#SBATCH --job-name="cmarkea"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=20:00:00

source /etc/profile.d/conda.sh
conda activate defi2

#Ajouter le programme que vous voulez lancer
SCRIPT_DIR="/users/kotmani/Defi2/models/Camembert"

python3 $SCRIPT_DIR/cmarkea.py

