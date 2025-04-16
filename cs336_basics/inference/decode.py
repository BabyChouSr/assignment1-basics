import argparse
import torch

from cs336_basics.layers.model import Model
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.layers.softmax import softmax

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEFAULT_TINYSTORIES_MODEL_CONFIG = {
    "vocab_size": 10000,
    "d_model": 512,
    "d_ff": 1344,
    "context_length": 256,
    "theta": 10000,
    "num_layers": 4,
    "num_heads": 16,
    "dtype": torch.float32,
}

DEFAULT_OWT_MODEL_CONFIG = {
    "vocab_size": 32000,
    "d_model": 512,
    "d_ff": 1344,
    "context_length": 256,
    "theta": 10000,
    "num_layers": 4,
    "num_heads": 16,
    "dtype": torch.float32,
}

def load_model(model_path):
    if "tinystories" in model_path:
        model = Model(**DEFAULT_TINYSTORIES_MODEL_CONFIG)
    elif "owt" in model_path:
        model = Model(**DEFAULT_OWT_MODEL_CONFIG)
    
    if device == "cpu":
        map_location = torch.device('cpu')
    else:
        map_location = None

    saved_model_weights = torch.load(model_path, map_location=map_location)["model_states"]
    print(saved_model_weights.keys())

    print(model)
    model.load_state_dict(saved_model_weights)

    return model

def apply_top_p(probabilities, top_p):
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probabilities, dim=-1)

    mask = cumulative_probs > top_p
    # breakpoint()
    mask = torch.zeros_like(probabilities, dtype=mask.dtype).scatter(dim=-1, index=sorted_indices, src=mask)
    masked_probabilities = probabilities.masked_fill(mask, 0)
    return masked_probabilities

def inference(model_path, 
              vocab_path,
              merges_path,
              prompt,
              max_new_tokens,
              temperature,
              top_p):
    model = load_model(model_path)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])

    tokens = torch.Tensor(tokenizer.encode(prompt)).unsqueeze(0).to(dtype=torch.long)
    
    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :]
        if temperature == 0:
            next_token = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            probabilities = softmax(logits, i=-1)
            probabilities = apply_top_p(probabilities, top_p)
            next_token = torch.multinomial(probabilities, num_samples=1)
        
        tokens = torch.cat([tokens, next_token], dim=1)

        if tokenizer.decode([next_token.item()]) == "<|endoftext|>":
            break

    tokens = tokens.squeeze(0)
    print(tokenizer.decode(tokens.tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--vocab-path", type=str)
    parser.add_argument("--merges-path", type=str)
    parser.add_argument("--prompt" , type=str)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type = float, default=1.0)

    args = parser.parse_args()

    inference(
        args.model_path,
        args.vocab_path,
        args.merges_path,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p
    )
