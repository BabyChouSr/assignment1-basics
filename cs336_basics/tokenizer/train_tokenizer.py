import argparse
import base64
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
    # WE use latin-1 because it maps the bytes byte to byte instead of utf-8 which
    # fails for certain bytes that require multiple bytes such as continuation bytes
    serializable_vocab = {k: v.decode('latin-1') for k, v in vocab.items()}

    with open(f"results/{args.dataset}-{args.split}-vocab.json", "w") as f:
        json.dump(serializable_vocab, f, indent=2)

    # Format merges for writing to file
    formatted_merges = []
    for first, second in merges:
        first_str = first.decode('latin-1')
        second_str = second.decode('latin-1')
        formatted_merges.append(f"({first_str},{second_str})")

    with open(f"results/{dataset_name}-merges.txt", "w") as f:
        f.write("\n".join(formatted_merges))
