#!/bin/bash
#SBATCH --job-name=owt_tokenized_valid
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=00:60:00
#SBATCH --output=slurm_outputs/owt_tokenize_%j.out
#SBATCH --error=slurm_outputs/owt_tokenize_%j.err

uv run python cs336_basics/tokenizer/tokenize_datasets.py --dataset /data/a1-basics/owt_valid.txt --output-path tokenized/owt-valid.npy --vocab-path tokenizers/owt-train-vocab.json --merges-path tokenizers/owt_train-merges.txt
