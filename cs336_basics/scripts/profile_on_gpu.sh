#!/bin/bash
#SBATCH --job-name=owt_train
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_outputs/train_owt_%j.out
#SBATCH --error=slurm_outputs/train_owt_%j.err

uv run python cs336_basics/profile_val_forward.py
