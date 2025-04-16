import submitit
import torch
from pathlib import Path

from cs336_basics.train.train import main

def run():
    print("Hello World")


default_tinystories_hyperparameters = {
        "vocab_size": 32000,
        "d_model": 512,
        "d_ff": 1344,
        "context_length": 256,
        "rope_theta": 10000,
        "num_layers": 4,
        "num_heads": 16,
        "num_train_tokens": 327680000,
        "max_lr": 0.004,
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
        "train_path": "tokenized/owt-train.npy",
        "validation_path": "tokenized/owt-train.npy",
        "batch_size": 256,
        "eval_batch_size": 512,
        "output_path": "models/owt-lr-",
        "wandb_entity": "babychousr-stanford-university",
        "wandb_project": "cs336-project1",
        "wandb_name": "owt-ctx-256-bsz-256-lr-",
        "model_type": None,
    }
learning_rate_sweeps = [0.004]

if __name__ == "__main__":
    folder = Path("slurm_outputs")
    folder.mkdir(exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder)


    for learning_rate in learning_rate_sweeps:
        parameters = default_tinystories_hyperparameters.copy()
        parameters["output_path"] = parameters["output_path"] + str(learning_rate)
        parameters["wandb_name"] = parameters["wandb_name"] + str(learning_rate)
        parameters["max_lr"] = learning_rate

        executor.update_parameters(
                timeout_min=120,
                slurm_partition="a1-batch",
                slurm_qos="a1-batch-qos",
                cpus_per_task=8,
                gpus_per_node=1,
                mem_gb=100,
                name=f"train_owt_lr_{learning_rate}",
            )
        
        job = executor.submit(main, **parameters)
        print(f"Submitted job ID: {job.job_id}")
