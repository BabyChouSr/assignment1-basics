import argparse
import json
from cs336_basics.tokenizer.train import train_bpe


DEFAULT_NUM_WORKERS = 8
DEFAULT_CHUNKSIZE = 25000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--dataset", type=str, default="TinyStoriesV2-GPT4")
    parser.add_argument("--vocab-size", type=int, default=10000)

    args = parser.parse_args()

    dataset_name = f"{args.dataset}-{args.split}"

    if "owt" in dataset_name:
        dataset_name = dataset_name.replace("-", "_")

    vocab, merges = train_bpe(f"data/{dataset_name}.txt", args.vocab_size, ["<|endoftext|>"], num_workers=DEFAULT_NUM_WORKERS, chunksize=DEFAULT_CHUNKSIZE)

    # Convert bytes objects to strings for JSON serialization
    serializable_vocab = {k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}

    with open(f"results/{args.dataset}-{args.split}-vocab.json", "w") as f:
        json.dump(serializable_vocab, f, indent=2)

    # Format merges for writing to file
    formatted_merges = []
    for first, second in merges:
        first_str = first.decode('utf-8', errors='replace')
        second_str = second.decode('utf-8', errors='replace')
        formatted_merges.append(f"({first_str},{second_str})")

    with open(f"results/{dataset_name}-merges.txt", "w") as f:
        f.write("\n".join(formatted_merges))
