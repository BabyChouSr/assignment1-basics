import argparse
import json
from cs336_basics.tokenizer.train import train_bpe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="valid")

    args = parser.parse_args()
    vocab, merges = train_bpe(f"data/TinyStoriesV2-GPT4-{args.split}.txt", 10000, ["<|endoftext|>"], num_workers=8)

    # Convert bytes objects to strings for JSON serialization
    serializable_vocab = {k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}

    with open(f"results/tinystories-{args.split}-vocab.json", "w") as f:
        json.dump(serializable_vocab, f, indent=2)

    # Format merges for writing to file
    # formatted_merges = []
    # for first, second in merges:
    #     first_str = first.decode('utf-8', errors='replace')
    #     second_str = second.decode('utf-8', errors='replace')
    #     formatted_merges.append(f"{first_str} {second_str}")

    # with open(f"results/merges.txt", "w") as f:
    #     f.write("\n".join(formatted_merges))
