import submitit
import torch
from pathlib import Path

from cs336_basics.inference.decode import inference

default_inference_hyperparameters = {
    "model_path": "models/tinystories-lr-0.004.pth",
    "vocab_path": "tokenizers/TinyStoriesV2-GPT4-train-vocab.json",
    "merges_path": "tokenizers/TinyStoriesV2-GPT4-train-merges.txt",
    "prompt": "Once upon a",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 1.0,
}

default_owt_hyperparameters = {
    "model_path": "models/owt-lr-0.004",
    "vocab_path": "tokenizers/owt-train-vocab.json",
    "merges_path": "tokenizers/owt_tarin-merges.txt",
    "prompt": "Once upon a",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 1.0,
}

if __name__ == "__main__":
    folder = Path("slurm_outputs")
    folder.mkdir(exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder)


    parameters = default_owt_hyperparameters.copy()
    executor.update_parameters(
            timeout_min=120,
            slurm_partition="a1-batch",
            slurm_qos="a1-batch-qos",
            cpus_per_task=8,
            gpus_per_node=1,
            mem_gb=100,
            name=f"infer_tinystories",
        )
    
    job = executor.submit(inference, **parameters)
    print(f"Submitted job ID: {job.job_id}")
