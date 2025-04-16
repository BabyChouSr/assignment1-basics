#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --error=/home/c-cychou/assignment1-basics/cs336_basics/scripts/slurm_outputs/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test_submitit
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/c-cychou/assignment1-basics/cs336_basics/scripts/slurm_outputs/%j_0_log.out
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH --signal=USR2@90
#SBATCH --time=120
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/c-cychou/assignment1-basics/cs336_basics/scripts/slurm_outputs/%j_%t_log.out --error /home/c-cychou/assignment1-basics/cs336_basics/scripts/slurm_outputs/%j_%t_log.err /scratch/c-cychou/uv-envs/assignment1-basics/bin/python3 -u -m submitit.core._submit /home/c-cychou/assignment1-basics/cs336_basics/scripts/slurm_outputs
