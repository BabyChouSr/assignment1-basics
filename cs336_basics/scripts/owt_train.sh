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

uv run torchrun cs336_basics/train/speedrun_train.py \
 --vocab-size 32000 \
 --d-model 512 \
 --d-ff 2048 \
 --context-length 512 \
 --rope-theta 10000 \
 --num-layers 6 \
 --num-heads 16 \
 --num-train-tokens 131072000 \
 --max-lr 0.004 \
 --min-lr 0.0004 \
 --lr-warmup-steps 250 \
 --checkpoint-steps 1000 \
 --validation-steps 1000 \
 --lr-scheduler-type cosine \
 --beta1 0.9 \
 --beta2 0.95 \
 --adam-eps 1e-8 \
 --weight-decay 0.01 \
 --train-path tokenized/owt-train.npy \
 --validation-path tokenized/owt-train.npy \
 --batch-size 128 \
 --eval-batch-size 256 \
 --output-path models/owt-speedrun-wd-0.01-cl-512-d-512-lr-4e-3 \
 --wandb-entity babychousr-stanford-university \
 --wandb-project cs336-project1 \
 --wandb-name owt-speedrun-wd-0.01-cl-512-d-512-4e-3-bsz-128-muon-0.02 \
 --use-muon True \
 --max-muon-lr 0.02

# 90 minutes probably: 851968000
# Then our C = roughly 3.83e16 = 852e6 * 45e6
# Change warmup steps too
# Set checkpoints step super large too
# 160 batch size using AdamW only, Muon must be lower
# 0.02 pretty good LR and faster - best so far
# Try RELU squared (d_ff 1344 -> 2048)
# Zero Init is faster by a little bit so keeping it
# Try increasing num layers and keep d_model