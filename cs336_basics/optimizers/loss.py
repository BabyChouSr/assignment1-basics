import torch

def cross_entropy(logits, targets):
    max_elem = torch.amax(logits, dim=-1, keepdim=True)
    logits = logits - max_elem

    exp_logits = torch.exp(logits)
    exp_sum = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_softmax = logits - torch.log(exp_sum)

    loss = -log_softmax.gather(-1, targets.unsqueeze(-1))
    loss = torch.mean(loss)
    return loss