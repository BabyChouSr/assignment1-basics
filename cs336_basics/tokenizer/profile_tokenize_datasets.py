import argparse
import concurrent.futures
import numpy as np
from functools import partial

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.train import find_chunk_boundaries

END_OF_TEXT_SPECIAL_TOKEN = "<|endoftext|>"
NUM_PROCESSES = 8


def encode_and_write_chunk(dataset, vocab_path: str, merges_path: str, special_tokens: list[str], boundaries):
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    chunk_size = 4096
    tokenized_array = []
    start, end = boundaries
    with open(dataset, 'rb') as f:
        f.seek(start)
        bytes_to_read = end - start
        while bytes_to_read > 0:
            read_size = min(chunk_size, bytes_to_read)
            chunk = f.read(read_size)
            for token_id in tokenizer.encode(chunk):
                tokenized_array.append(token_id)
            bytes_to_read -= len(chunk)

    return tokenized_array
    

def main(args):
    special_tokens = [END_OF_TEXT_SPECIAL_TOKEN]

    with open(args.dataset, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, END_OF_TEXT_SPECIAL_TOKEN.encode("utf-8"))

    iteration_pairs = list(zip(boundaries[:-1], boundaries[1:]))
    process_tokenize_func = partial(encode_and_write_chunk, dataset=args.dataset, vocab_path=args.vocab_path, merges_path=args.merges_path, special_tokens=special_tokens)
    np_tokenized = np.array()
    with concurrent.futures.ProcessPoolExecutor(num_workers=NUM_PROCESSES) as executor:
        results = executor.map(process_tokenize_func, iteration_pairs)

        for tokenized_array in results:
            np_tokenized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--vocab-path", default="results/TinyStoriesV2-GPT4-train-vocab.json")
    parser.add_argument("--merges-path", default="results/TinyStoriesV2-GPT4-train-merges.txt")
    parser.add_argument("--")
    args = parser.parse_args()

    main(args)
