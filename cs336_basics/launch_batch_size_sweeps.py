import submitit
import torch
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
        "lr_warmup_steps": 500,
        "checkpoint_steps": 1000,
        "validation_steps": 1000,
        "lr_scheduler_type": "cosine",
        "beta1": 0.9,
        "beta2": 0.999,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "dtype": torch.float32,
        "train_path": "tokenized/tinystories-train.npy",
        "validation_path": "tokenized/tinystories-train.npy",
        "batch_size": 256,
        "eval_batch_size": 1024,
        "output_path": "models/tinystories-lr-0.0001-",
        "wandb_entity": "babychousr-stanford-university",
        "wandb_project": "cs336-project1",
        "wandb_name": "tinystories-ctx-256-lr-bsz-",
        "model_type": None
    }
batch_size_sweeps = [1024, 2048, 4096]

if __name__ == "__main__":
    folder = Path("slurm_outputs")
    folder.mkdir(exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder)


    for batch_size in batch_size_sweeps:
        parameters = default_tinystories_hyperparameters.copy()
        parameters["output_path"] = parameters["output_path"] + str(batch_size)
        parameters["wandb_name"] = parameters["wandb_name"] + str(batch_size)
        parameters["batch_size"] = batch_size
        executor.update_parameters(
                timeout_min=120,
                slurm_partition="a1-batch",
                slurm_qos="a1-batch-qos",
                cpus_per_task=8,
                gpus_per_node=1,
                mem_gb=100,
                name=f"train_tinystories_lr_{batch_size}",
            )
        
        job = executor.submit(main, **parameters)
        print(f"Submitted job ID: {job.job_id}")
