#!/bin/sh
#SBATCH --job-name=hydrate_compare
#SBATCH --account=share-nv-fys
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

module load CUDA/12.4.0 Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate nect

python compare_dynamic.py scans/2026-11-23_173036 outputs/2026-11-23_173036 --video
