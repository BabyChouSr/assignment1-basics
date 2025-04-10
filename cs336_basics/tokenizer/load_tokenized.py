import argparse
import numpy as np
import os
from cs336_basics.tokenizer.tokenizer import Tokenizer

def load_tokenized_dataset(file_path, num_tokens=1000):
    """
    Load a tokenized dataset from a numpy file using memory mapping.
    
    Args:
        file_path: Path to the numpy file containing tokenized data
        num_tokens: Number of tokens to load for preview
        
    Returns:
        Tuple of (memory-mapped array, preview tokens)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Memory-map the file
    tokens = np.load(file_path, mmap_mode='r')
    
    # Get a preview of the first num_tokens
    preview = tokens[:min(num_tokens, len(tokens))].copy()
    # preview = [int(token) for token in preview]
    suffix = tokens[-min(num_tokens, len(tokens)):].copy()

    tokenizer = Tokenizer.from_files("results/TinyStoriesV2-GPT4-train-vocab.json", "results/TinyStoriesV2-GPT4-train-merges.txt", ["<|endoftext|>"])

    print(f"Loaded dataset from {file_path}")
    print(f"Total tokens: {len(tokens)}")
    print(f"First {len(preview)} tokens: {preview}")

    text = tokenizer.decode(preview)
    print(text)

    print("=" * 20 + "SUFFIX TEXT" + "=" * 20)

    suffix_text = tokenizer.decode(suffix)
    print(suffix_text)
    
    return tokens, preview

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a tokenized dataset from a numpy file")
    parser.add_argument("--dataset", type=str, default="tokenized/tinystories-valid.npy",
                        help="Path to the tokenized dataset file")
    parser.add_argument("--num-tokens", type=int, default=1000,
                        help="Number of tokens to load for preview")
    
    args = parser.parse_args()
    
    tokens, preview = load_tokenized_dataset(args.dataset, args.num_tokens)