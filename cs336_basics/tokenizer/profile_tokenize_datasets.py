import argparse
import time

from cs336_basics.tokenizer.tokenizer import Tokenizer

def main(args):
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=special_tokens)

    text = ""
    end_of_text_count = 0
    with open(args.dataset, "r") as f:
        for line in f:
            text += line
            end_of_text_count += line.count("<|endoftext|>")

            if end_of_text_count >= 10:
                break

    start = time.time()
    ids = tokenizer.encode(text)
    end = time.time()
    byte_array = text.encode("utf-8")
    print(f"Number of tokens: {len(ids)}")
    print(f"Number of bytes: {len(byte_array)}")
    print(f"Compression ratio: {len(byte_array)/len(ids)}")
    print(f"Time: {end - start}")
    print(f"Throughput (bytes / s): {len(byte_array) / (end - start)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--vocab-path", default="results/TinyStoriesV2-GPT4-train-vocab.json")
    parser.add_argument("--merges-path", default="results/TinyStoriesV2-GPT4-train-merges.txt")
    args = parser.parse_args()

    main(args)

