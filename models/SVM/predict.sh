#!/bin/bash
#SBATCH --job-name="reader"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=20:00:00

source /etc/profile.d/conda.sh
conda activate defi2

#Ajouter le programme que vous voulez lancer
SCRIPT_DIR="/users/kotmani/Defi2/models/SVM/"

python3 $SCRIPT_DIR/readSVM_predictions.py
