import torch
import numpy as np

np.random.seed(42)

def get_batch(x: np.array, batch_size: int, context_length: int, device: str):
    possible_choices = len(x) - context_length
    tensor = torch.from_numpy(x)
    random_start_indices = torch.randint(0, possible_choices, (batch_size,))
    
    # Create indices for each sequence in the batch
    offsets = torch.arange(0, context_length)
    
    # Use broadcasting to create indices for all sequences at once
    # This creates a batch_size x context_length tensor of indices
    # (bsz, 1) + (1, ctx_len)
    indices = random_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
    
    # Get batch and targets using the indices
    batch = torch.index_select(tensor, 0, indices.view(-1)).view(batch_size, context_length).to(device=device)
    target = torch.index_select(tensor, 0, (indices + 1).view(-1)).view(batch_size, context_length).to(device=device)
    
    return batch, target