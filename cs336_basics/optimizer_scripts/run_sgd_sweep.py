import torch
import torch.nn
from cs336_basics.optimizers.sgd import SGD


def learning_rate_tuning(lr):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.


lrs = [1e1, 1e2, 1e3]
for lr in lrs:
    print(f"Starting learning rate sweep with lr: {lr}")
    learning_rate_tuning(lr)