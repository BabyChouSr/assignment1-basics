import submitit
from pathlib import Path

from cs336_basics.train.train import main

def run():
    print("Hello World")


default_tinystories_hyperparameters = {
        "vocab_size": 10000,
        "d_model": 512,
        "d_ff": 1344,
        "context_length": 256,
        "rope_theta": 10000,
        "num_layers": 4,
        "num_heads": 16,
        "num_train_tokens": 327680000,
        "max_lr": 0.0001,
        "min_lr": 0,
        "lr_scheduler_type": "cosine",
        "beta1": 0.9,
        "beta2": 0.999,
        "train_path": "tokenized/tinystories-train.npy",
        "validation_path": "tokenized/tinystories-train.npy",
        "batch_size": 256,
        "eval_batch_size": 1024,
        "output_path": "models/tinystories-lr-",
        "wandb_name": "tinystories-ctx-256-bsz-256-lr-",
    }



if __name__ == "__main__":
    folder = Path("slurm_outputs")
    folder.mkdir(exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(
            timeout_min=120,
            slurm_partition="a1-batch-cpu",
            slurm_qos="a1-batch-cpu-qos",
            cpus_per_task=8,
            gpus_per_node=0,
            mem_gb=100,
            name="test_submitit",
        )
    
    job = executor.submit(run)
    print(f"Submitted job ID: {job.job_id}")
