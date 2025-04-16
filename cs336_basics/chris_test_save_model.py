import torch
from cs336_basics.layers.model import Model
from cs336_basics.inference.decode import DEFAULT_TINYSTORIES_MODEL_CONFIG
from cs336_basics.optimizers.adamw import AdamW
from cs336_basics.train.checkpoint import load_checkpoint, save_checkpoint

checkpoint_data = {}
model = Model(**DEFAULT_TINYSTORIES_MODEL_CONFIG)
model = torch.compile(model)
tokens = torch.Tensor([123, 123, 123]).to(dtype=torch.long).unsqueeze(0)
model(tokens)
# optimizer = AdamW(model.parameters())
# save_checkpoint(model, optimizer, 0, "models/test")
# saved_info = torch.load("models/test")
# print(saved_info["model_states"].keys())