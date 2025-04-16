import torch
import numpy as np

from cs336_basics.model.speedrun_model import SpeedrunModel
from cs336_basics.inference.decode import DEFAULT_OWT_MODEL_CONFIG
from cs336_basics.optimizers.loss import cross_entropy
from cs336_basics.optimizers.adamw import AdamW
from cs336_basics.train.data_loader import get_batch
from cs336_basics.optimizers.lr_scheduler import gradient_clipping

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def calculate_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm = p.grad.data.norm(2).item()
            total_norm += grad_norm ** 2
    
    total_norm = total_norm ** 0.5
    return total_norm

checkpoint_data = {}

batch_size = 180
context_length = 512

train_tokens = np.load("tokenized/owt-valid.npy", mmap_mode="r+")
x, y = get_batch(train_tokens, batch_size, context_length, device)
config = DEFAULT_OWT_MODEL_CONFIG.copy()
config["context_length"] = 512
model = SpeedrunModel(**config).to(device)
# model = torch.compile(model)
optimizer = AdamW(model.parameters())

scaler = torch.amp.GradScaler(device)
# loss_fct = torch.nn.CrossEntropyLoss()
loss_fct = cross_entropy
with torch.amp.autocast(device, dtype=torch.bfloat16):
    # loss = cross_entropy(model(x), y)
    # outputs = model(x)
    # outputs = model(x).view(x.size(0), -1)
    # print(outputs.shape)
    loss = loss_fct(model(x).view(batch_size * context_length, -1), y.view(batch_size * context_length))

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
# grad_norm = calculate_grad_norm(model)
print(calculate_grad_norm(model))
grad_norm = gradient_clipping(model.parameters(), 3)
print(grad_norm)

    
