import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

from cs336_basics.train.data_loader import get_batch
from cs336_basics.layers.model import Model
from cs336_basics.optimizers.loss import cross_entropy

validation_path = "tokenized/owt-valid.npy"
validation_data = np.load(validation_path, mmap_mode="r+")
context_length = 256
eval_batch_size = 10
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(
    d_model=default_tinystories_hyperparameters["d_model"],
    num_heads=default_tinystories_hyperparameters["num_heads"],
    d_ff=default_tinystories_hyperparameters["d_ff"],
    theta=default_tinystories_hyperparameters["rope_theta"],
    vocab_size=default_tinystories_hyperparameters["vocab_size"],
    context_length=default_tinystories_hyperparameters["context_length"],
    num_layers=default_tinystories_hyperparameters["num_layers"],
    device=device,
    dtype=default_tinystories_hyperparameters["dtype"]
).to(device)

model.eval()


print("start profiler")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    try:
        for i in range(5):
            with record_function("data load"):
                x, y = get_batch(validation_data, eval_batch_size, context_length, device)
            with torch.no_grad():
                with record_function("model_inference"):
                    logits = model(x)
                
                with record_function("loss"):
                    loss = cross_entropy(logits, y)
                print(f"Sample {i+1} loss: {loss.item()}")
    except Exception as e:
        print(f"exception during profiling: {e}")

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))
    