#!/bin/bash
#SBATCH --job-name=test_hello_batch
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH --time=00:05:00
#SBATCH --output=hello_batch_%j.out
#SBATCH --error=hello_batch_%j.err

uv run hello_batch.py

