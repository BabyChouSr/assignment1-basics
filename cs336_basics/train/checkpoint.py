import torch

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint_data = {}
    checkpoint_data["model_states"] = model.state_dict()
    checkpoint_data["optimizer_states"] = optimizer.state_dict()
    checkpoint_data["iteration_number"] = iteration

    torch.save(checkpoint_data, out)

def load_checkpoint(src, model, optimizer):
    checkpoint_data = torch.load(src)
    model.load_state_dict(checkpoint_data["model_states"])
    optimizer.load_state_dict(checkpoint_data["optimizer_states"])

    return checkpoint_data["iteration_number"]