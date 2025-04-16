import argparse
import os
import wandb
import numpy as np
import torch
import time

from cs336_basics.model.speedrun_model import SpeedrunModel
from cs336_basics.optimizers.adamw import AdamW
from cs336_basics.train.data_loader import get_batch
from cs336_basics.train.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.optimizers.loss import cross_entropy
from cs336_basics.optimizers.lr_scheduler import lr_scheduler_step, lr_cosine_schedule, gradient_clipping

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def eval_loss_loop(
    model,
    validation_data,
    context_length,
    eval_batch_size,
):
    total_loss = 0
    # num_samples = (len(validation_data) - context_length) // (eval_batch_size * context_length)
    num_samples = 100
    model.eval()
    with torch.no_grad():
        # for i in range(num_samples):
        #     start_idx = i * (eval_batch_size * context_length)
            
        #     data = torch.Tensor(validation_data[start_idx: start_idx + eval_batch_size * context_length]).reshape(eval_batch_size, context_length).to(dtype=torch.long, device=device)
        #     targets = torch.Tensor(validation_data[start_idx + 1: start_idx + eval_batch_size * context_length + 1]).reshape(eval_batch_size, context_length).to(dtype=torch.long, device=device)
        #     loss = cross_entropy(model(data), targets)
        #     total_loss += loss.item()

        for _ in range(num_samples):
            x, y = get_batch(validation_data, eval_batch_size, context_length, device)
            loss = cross_entropy(model(x), y)
            total_loss += loss.item()

    model.train()

    return total_loss / num_samples


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
    eval_batch_size,
    checkpoint_steps,
    validation_steps,
    output_path,
    wandb_entity,
    wandb_project,
    wandb_name,
    model_type: str | None = None,
    max_grad_norm: int = 3,
    use_muon: bool = False,
    max_muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
):

    device_id = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl", device_id=device_id)
    torch.distributed.barrier()
    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=wandb_name,
        config=locals(),
        resume="allow",
    )

    if model_type is None:
        model_cls = SpeedrunModel

    if model_type in ["nope"]:
        model = model_cls(d_model, num_heads, d_ff, vocab_size, context_length, num_layers, device, dtype)
    else:
        model = model_cls(d_model, num_heads, d_ff, rope_theta, vocab_size, context_length, num_layers, device, dtype)

    if not use_muon:
        optimizer = AdamW(model.parameters(), max_lr, (beta1, beta2), adam_eps, weight_decay)
        optimizers = [optimizer]
        trainable_params = sum([p.numel() for p in model.parameters()])
    else:
        from muon import Muon

        muon_params = [p for p in model.layers.parameters() if p.ndim >= 2]
        adamw_params = ([p for p in model.layers.parameters() if p.ndim < 2]
              + [*model.token_embeddings.parameters(), *model.lm_head.parameters(), *model.ln_final.parameters()])
            
        optimizers = [Muon(muon_params, lr=max_muon_lr, momentum=muon_momentum, rank=0, world_size=1),
              AdamW(adamw_params, lr=max_lr, betas=(beta1, beta2), weight_decay=weight_decay)] 

        trainable_params = sum([p.numel() for p in muon_params]) + sum([p.numel() for p in adamw_params])
    if os.path.exists(output_path):
        try:
            start_step = load_checkpoint(output_path, model, optimizer)
            print(f"Found checkpoint and resuming at step: {start_step}")
        except:
            start_step = 0
    else:
        start_step = 0


    print(f"Number of trainable parameters: {trainable_params}")

    # NOTE(Chris): not working atm because of attention mask or something - fixed use float16 instead of bool
    # NOTE(chris): This doesn't work with mixed precision training for some reason
    # model = torch.compile(model)

    train_tokens = np.load(train_path, mmap_mode="r")
    validation_tokens = np.load(validation_path, mmap_mode="r")

    num_train_steps = int(num_train_tokens / (context_length * batch_size))
    start_time = time.time()

    print(f"Training the model for {num_train_tokens} tokens => {num_train_steps} steps")

    scaler = torch.amp.GradScaler(device)
    for step in range(start_step, num_train_steps):
        # Putting this first because we want to set the lr in each of the optimizer groups corretly.
        # We do step + 1 because we start at t = 1 for AdamW optimizer
        for optimizer in optimizers:
            if isinstance(optimizer, AdamW):
                lr_scheduler_step(optimizer, lr_scheduler_type, step + 1, max_lr, min_lr, lr_warmup_steps, num_train_steps)
            elif isinstance(optimizer, Muon):
                lr_scheduler_step(optimizer, lr_scheduler_type, step + 1, max_muon_lr, max_muon_lr * 0.10, lr_warmup_steps, num_train_steps)

        x, y = get_batch(train_tokens, batch_size, context_length, device)

        for optimizer in optimizers:
            optimizer.zero_grad()
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            loss = cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
            grad_norm = gradient_clipping(model.parameters(), max_grad_norm)
            scaler.step(optimizer)

        scaler.update()

        run.log({"train/loss": loss, "train/learning_rate": lr_cosine_schedule(step + 1, max_lr, min_lr, lr_warmup_steps, num_train_steps), "train/grad": grad_norm, "train/seconds": time.time() - start_time})

        if step % validation_steps == 0:
            validation_loss = eval_loss_loop(model, validation_tokens, context_length, eval_batch_size)
            run.log({"eval/loss": validation_loss})
        
        if step % checkpoint_steps == 0:
            save_checkpoint(model, optimizer, step, output_path)

    validation_loss = eval_loss_loop(model, validation_tokens, context_length, eval_batch_size)
    run.log({"eval/loss": validation_loss})

    save_checkpoint(model, optimizer, step, output_path)
    torch.save(model.state_dict(), f"{output_path}-weights")

    run.finish()

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
    parser.add_argument("--min-lr", type=float, default=1e-5)
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
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--wandb-entity", type=str, default="babychousr-stanford-university")
    parser.add_argument("--wandb-project", type=str, default="cs336-project1")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=3)
    parser.add_argument("--use-muon", type=bool, default=False)
    parser.add_argument("--max-muon-lr", type=float, default=0.002)
    parser.add_argument("--muon-momentum", type=float, default=0.95)

    args = parser.parse_args()

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
        eval_batch_size=args.eval_batch_size,
        checkpoint_steps=args.checkpoint_steps,
        validation_steps=args.validation_steps,
        output_path=args.output_path,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        model_type=args.model_type,
        max_grad_norm=args.max_grad_norm,
        use_muon=args.use_muon,
        max_muon_lr=args.max_muon_lr,
        muon_momentum=args.muon_momentum,
    )