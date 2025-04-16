#!/bin/bash
#SBATCH --job-name=tinystories_tokenized_train
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=00:60:00
#SBATCH --output=slurm_outputs/tinystories_tokenize_%j.out
#SBATCH --error=slurm_outputs/tinystories_tokenize_%j.err

uv run python cs336_basics/tokenizer/tokenize_datasets.py --dataset /data/a1-basics/TinyStoriesV2-GPT4-valid.txt --output-path tokenized/tinystories-valid.npy --vocab-path tokenizers/TinyStoriesV2-GPT4-train-vocab.json --merges-path tokenizers/TinyStoriesV2-GPT4-train-merges.txt
