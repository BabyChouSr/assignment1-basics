import argparse
import concurrent.futures
import numpy as np
from functools import partial

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.train import find_chunk_boundaries

END_OF_TEXT_SPECIAL_TOKEN = "<|endoftext|>"
NUM_PROCESSES = 8


def encode_and_write_chunk(boundaries, dataset, vocab_path: str, merges_path: str, special_tokens: list[str]):
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    start, end = boundaries
    endoftext_token = b"<|endoftext|>"
    token_length = len(endoftext_token)
    buffer_size = 4096  # Read size in bytes

    tokens = []

    # with open(dataset, "rb") as f:
    #     f.seek(start)
    #     text = f.read(end - start).decode("utf-8", errors="ignore")
    #     return tokenizer.encode(text)

    with open(dataset, "rb") as f:
        f.seek(start)
        current_position = start
        
        # Keep reading until we reach the end position
        while current_position < end:
            # print(current_position)
            # Read a document up to the next <|endoftext|> token
            document = b""
            token_found = False
            
            while not token_found and current_position < end:
                # Determine how much to read
                remaining = min(buffer_size, end - current_position)
                if remaining <= 0:
                    break
                    
                # Read a chunk
                chunk = f.read(remaining)
                if not chunk:  # End of file
                    break
                    
                # Look for the endoftext token
                token_pos = chunk.find(endoftext_token)
                
                if token_pos != -1:
                    # Found token in this chunk
                    document += chunk[:token_pos + token_length]  # Add text up to token
                    token_found = True
                    
                    # Update position to after the token
                    current_position += token_pos + token_length
                    f.seek(current_position)
                else:
                    # No token in this chunk, keep looking
                    document += chunk
                    current_position += len(chunk)
                        
            # Process the document if we have content
            if document:
                try:
                    text = document.decode("utf-8", errors="ignore")
                    tokens.extend(tokenizer.encode(text))
                except Exception as e:
                    print(f"Error processing document: {e}")
    
    return tokens
    

def main(args):
    special_tokens = [END_OF_TEXT_SPECIAL_TOKEN]

    with open(args.dataset, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, END_OF_TEXT_SPECIAL_TOKEN.encode("utf-8"))

    iteration_pairs = list(zip(boundaries[:-1], boundaries[1:]))
    process_tokenize_func = partial(encode_and_write_chunk, dataset=args.dataset, vocab_path=args.vocab_path, merges_path=args.merges_path, special_tokens=special_tokens)
    np_tokenized = np.array([], dtype=np.uint16)
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        results = executor.map(process_tokenize_func, iteration_pairs)

        for tokenized_array in results:
            np_tokenized = np.append(np_tokenized, tokenized_array)

    np.save(args.output_path, np_tokenized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--vocab-path", default="results/TinyStoriesV2-GPT4-train-vocab.json")
    parser.add_argument("--merges-path", default="results/TinyStoriesV2-GPT4-train-merges.txt")
    parser.add_argument("--output-path", default="tokenized/tinystories.npy")
    args = parser.parse_args()

    main(args)
