#!/bin/bash
#SBATCH --job-name=owt_tokenized_train
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/owt_tokenize_%j.out
#SBATCH --error=slurm_outputs/owt_tokenize_%j.err

uv run python cs336_basics/tokenizer/tokenize_datasets.py --dataset /data/a1-basics/owt_train.txt --output-path tokenized/owt-train.npy --vocab-path tokenizers/owt-train-vocab.json --merges-path tokenizers/owt_train-merges.txt
