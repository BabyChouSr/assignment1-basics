import torch
from typing import Callable, Optional
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate
            beta1, beta2= group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                moment = state.get("moment", 0)
                second_moment = state.get("second_moment", 0)

                grad = p.grad.data
                moment = beta1 * moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * grad**2

                a_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= a_t * moment / (torch.sqrt(second_moment) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["moment"] = moment
                state["second_moment"] = second_moment

        return loss
