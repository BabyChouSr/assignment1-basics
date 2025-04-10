import argparse
import wandb
import numpy as np
import torch

from cs336_basics.layers.model import Model
from cs336_basics.optimizers.adamw import AdamW
from cs336_basics.train.data_loader import get_batch
from cs336_basics.train.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.optimizers.loss import cross_entropy
from cs336_basics.optimizers.lr_scheduler import lr_scheduler_step, lr_cosine_schedule

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(
    vocab_size,
    context_length,
    d_model,
    d_ff,
    rope_theta,
    num_layers,
    num_heads,
    num_train_tokens,
    max_lr,
    min_lr,
    lr_scheduler_type,
    lr_warmup_steps,
    beta1,
    beta2,
    adam_eps,
    weight_decay,
    dtype,
    train_path,
    validation_path,
    batch_size,
    checkpoint_steps,
    validation_steps,
    wandb_run,
):
    model = Model(d_model, num_heads, d_ff, rope_theta, vocab_size, context_length, num_layers, device, dtype)
    optimizer = AdamW(model.parameters(), max_lr, (beta1, beta2), adam_eps, weight_decay)

    print(f"Number of trainable parameters: {sum([p.numel() for p in model.parameters()])}")

    # NOTE(Chris): not working atm because of attention mask or something
    # model = torch.compile(model)

    train_tokens = np.load(train_path, mmap_mode="r")
    validation_tokens = np.load(validation_path, mmap_mode="r")

    num_train_steps = int(num_train_tokens / (context_length * batch_size))

    print(f"Training the model for {num_train_tokens} tokens => {num_train_steps} steps")

    for step in range(num_train_steps):
        # Putting this first because we want to set the lr in each of the optimizer groups corretly.
        # We do step + 1 because we start at t = 1 for AdamW optimizer
        lr_scheduler_step(optimizer, lr_scheduler_type, step + 1, max_lr, min_lr, lr_warmup_steps, num_train_steps)

        x, y = get_batch(train_tokens, batch_size, context_length, device)
        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

        wandb_run.log({"train/loss": loss, "train/learning_rate": lr_cosine_schedule(step + 1, max_lr, min_lr, lr_warmup_steps, num_train_steps)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=int, default=10000)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-train-tokens", type=int, default=327680000)
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=0)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float32)
    parser.add_argument("--train-path", type=str)
    parser.add_argument("--validation-path", type=str)
    parser.add_argument("--checkpoint-steps", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--wandb-entity", type=str, default="babychousr-stanford-university")
    parser.add_argument("--wandb-project", type=str, default="cs336-project1")
    parser.add_argument("--wandb-name", type=str)

    args = parser.parse_args()

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=vars(args)
    )

    main(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_train_tokens=args.num_train_tokens,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_warmup_steps=args.lr_warmup_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
        dtype=args.dtype,
        train_path=args.train_path,
        validation_path=args.validation_path,
        batch_size=args.batch_size,
        checkpoint_steps=args.checkpoint_steps,
        validation_steps=args.validation_steps,
        wandb_run=run,
    )

    run.finish()