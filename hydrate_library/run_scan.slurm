#!/bin/sh
#SBATCH --job-name=hydrate_scan
#SBATCH --account=aurorsta      # e.g. the account you use for your other IDUN jobs
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1                       # add a GPU type here if you need one, as in your other jobs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Same module loads as when you installed NeCT (if you used any), then the NeCT environment
module load CUDA/12.4.0 Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate nect

# First run: python scan_gpu.py config.yaml --static     (checks that the geometry matches NeCT)
python scan_gpu.py configs.yaml
