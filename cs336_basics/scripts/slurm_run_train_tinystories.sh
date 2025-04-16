#!/bin/bash
#SBATCH --job-name=tinystories_train
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/train_tinystories_%j.out
#SBATCH --error=slurm_outputs/train_tinystories_%j.err

uv run python cs336_basics/train/train.py --vocab-size 10000 \
 --context-length 256 \
 --d-model 512 \
 --d-ff 1344 \
 --rope-theta 10000 \
 --num-layers 4 \
 --num-heads 16 \
 --num-train-tokens 327680000 \
 --max-lr 0.0001 \
 --min-lr 0 \
 --lr-scheduler-type cosine \
 --beta1 0.9 \
 --beta2 0.999 \
 --train-path tokenized/tinystories-train.npy \
 --validation-path tokenized/tinystories-valid.npy \
 --batch-size 512 \
 --eval-batch-size 1024 \
 --output-path models/tinystories-lr-1e-4 \
 --wandb-name tinystories-lr-1e-4-ctx-256-bsz-512
