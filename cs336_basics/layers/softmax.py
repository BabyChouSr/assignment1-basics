import torch

def softmax(x: torch.Tensor, i: int):
    x_max = torch.amax(x, dim=i, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=i, keepdim=True)
    return x_exp / x_exp_sum
