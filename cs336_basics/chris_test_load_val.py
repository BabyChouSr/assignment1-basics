import numpy as np
import torch

validation_path = "tokenized/owt-valid.npy"
validation_data = np.load(validation_path, mmap_mode="r")
context_length = 256
eval_batch_size = 512

num_samples = (len(validation_data) - context_length) // (eval_batch_size * context_length)
for i in range(num_samples):
    start_idx = i * (eval_batch_size * context_length)
    data = torch.Tensor(validation_data[start_idx: start_idx + eval_batch_size * context_length]).view(eval_batch_size, context_length).to(dtype=torch.long)
    targets = torch.Tensor(validation_data[start_idx + 1: start_idx + eval_batch_size * context_length + 1]).view(eval_batch_size, context_length).to(dtype=torch.long)
