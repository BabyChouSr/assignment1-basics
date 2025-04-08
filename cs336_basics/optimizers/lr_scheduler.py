import math
import torch

def lr_cosine_schedule(t, max_lr, min_lr, warmup_steps, total_steps):
    if t < warmup_steps:
        return t / warmup_steps * max_lr
    elif t >= warmup_steps and t <= total_steps:
        return min_lr + 0.5 * (1 + math.cos((1 / (total_steps - warmup_steps) * (t - warmup_steps) * math.pi))) * (max_lr - min_lr)
    else:
        return min_lr
    
def gradient_clipping(parameters, max_l2_norm):
    eps = 1e-6
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            grad_norm = p.grad.data.norm(2).item()
            total_norm += grad_norm ** 2
    
    total_norm = total_norm ** 0.5

    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

