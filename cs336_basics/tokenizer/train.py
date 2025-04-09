import os
from typing import BinaryIO

import itertools
import regex as re
import concurrent.futures
from functools import partial
from collections import defaultdict, Counter

INITIAL_BYTE_VOCAB_SIZE = 256
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
END_OF_TEXT_TOKEN = "<|endoftext|>"


def char_to_bytes(s):
    return bytes(s, encoding="utf-8")

def str_to_byte_list(s):
    return [bytes([b]) for b in s.encode("utf-8")]

# def negative_ord_tuple(pair):
#     first = tuple([-byte for byte in pair[0]])
#     second = tuple([-byte for byte in pair[1]])
#     return (first, second)

# def positive_bytes(neg_ord_tuple):
#     first, second = neg_ord_tuple
#     first = bytes([-byte for byte in first])
#     second = bytes([-byte for byte in second])
#     return (first, second)

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]  # Chunks start on previous index, don't include last index
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
            if mini_chunk == b"":  # If EOF, this boundary should be at the end of the file
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)  # Find the special token in the mini chunk
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def lookup_pair_counts_and_locations(
    subwords_and_offset: tuple,
    subword_freqs: dict,
):
    subwords, offset = subwords_and_offset
    pair_locations = defaultdict(list)
    pair_counts = Counter()

    for word_idx, word in enumerate(subwords):
        word_freq = subword_freqs[tuple(word)]
        for pos in range(len(word) - 1):
            pair = (word[pos], word[pos + 1])
            pair_counts[pair] += word_freq
            pair_locations[pair].append(word_idx + offset)

    return pair_counts, pair_locations

def build_subword_freq_table(
    documents: list[str],
):
    subword_freqs = {}
    for document in documents:
        # Special token used for delimiting endoftext should be removed
        document = document.replace("<|endoftext|>", "")
        subword_iter = re.finditer(PAT, document)
        for subword_match in subword_iter:
            subword_str = subword_match.group()
            # subword = [char_to_bytes(c) for c in subword_str]
            subword = str_to_byte_list(subword_str)
            subword_byte_tuple = tuple(subword)
            subword_freqs[subword_byte_tuple] = subword_freqs.get(subword_byte_tuple, 0) + 1
    
    return subword_freqs


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers=1,
    chunksize=25000,
):
    num_merges = vocab_size - len(special_tokens) - INITIAL_BYTE_VOCAB_SIZE
    vocab = {}
    for i in range(INITIAL_BYTE_VOCAB_SIZE):
        vocab[i] = bytes([i])

    for i, special_token in enumerate(special_tokens):
        vocab[INITIAL_BYTE_VOCAB_SIZE + i] = bytes(special_token, encoding="utf-8")

    # with open(input_path) as f:
    #     corpus_text = str(f.read())
    #     documents = corpus_text.split(END_OF_TEXT_TOKEN)

    # batch_size = max(1, len(documents) // num_workers)
    # batches = [documents[i: i + batch_size] for i in range(0, len(documents), batch_size)]

    subword_freqs = Counter()
    subwords = set()

    # Memory optimization by doing streaming
    with open(input_path, 'r') as f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Process file in chunks
            while True:
                # Read a chunk of lines
                lines_chunk = list(itertools.islice(f, chunksize))
                if not lines_chunk:
                    break
                    
                # Divide chunk into smaller batches for workers
                batch_size = max(1, len(lines_chunk) // num_workers)
                batches = [lines_chunk[i:i+batch_size] for i in range(0, len(lines_chunk), batch_size)]
                
                # Process batches in parallel
                results = executor.map(build_subword_freq_table, batches)
                
                # Combine results
                for local_subword_freqs in results:
                    for word, freq in local_subword_freqs.items():
                        subword_freqs[word] += freq
                        subwords.add(word)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     results = executor.map(build_subword_freq_table, batches)

    #     for local_subword_freqs in results:
    #         for word, freq in local_subword_freqs.items():
    #             subword_freqs[word] += freq
    #             subwords.add(word)
    
    subwords = list(subwords)

    
    # subword_iter = re.finditer(PAT, corpus_text)

    # subword_freqs = {}
    # for subword_match in subword_iter:
    #     subword_str = subword_match.group()
    #     subword = [char_to_bytes(c) for c in subword_str]
    #     subword_byte_tuple = tuple(subword)
    #     subword_freqs[subword_byte_tuple] = subword_freqs.get(subword_byte_tuple, 0) + 1

    # subwords = list(subword_freqs.keys())

    pair_locations = defaultdict(list)
    pair_counts = Counter()

    # Calculate batch size - divide work evenly among workers
    batch_size = max(1, len(subwords) // num_workers)
    batches = [(subwords[i : i + batch_size], i) for i in range(0, len(subwords), batch_size)]

    process_batch = partial(lookup_pair_counts_and_locations, subword_freqs=subword_freqs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_batch, batches)

        # Combine results from all workers
        for local_counts, local_locations in results:
            for pair, locations in local_locations.items():
                pair_locations[pair].extend(locations)
                pair_counts[pair] += local_counts[pair]

    # print(pair_counts)
    # subwords = list(subword_freqs.keys())
    # for word_idx, word in enumerate(subwords):
    #     word_freq = subword_freqs[tuple(word)]
    #     for pos in range(len(word) - 1):
    #         pair = (word[pos], word[pos + 1])
    #         pair_counts[pair] += word_freq
    #         pair_locations[pair].append(word_idx)

    # Max heap on counts and lexicographically greater pairs
    # Sort and print pair_counts from largest to smallest count

    # NOTE(chris): UTIL TO print sorted
    # sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    # print("Pairs sorted by frequency (highest to lowest):")
    # for pair, count in sorted_pairs:
    #     print(f"Pair: {positive_bytes(negative_ord_tuple(pair))}, Count: {count}")

    # pair_pq = [(-counts, negative_ord_tuple(pair)) for pair, counts in pair_counts.items()]
    # heapq.heapify(pair_pq)

    merges = []
    for j in range(num_merges):
        # if not pair_pq:
        #     break

        if len(pair_counts) == 0:
            break

        # if j in list(range(190, 198)):
        #     sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        #     print(f"Pairs sorted by frequency (highest to lowest) for iteraiton {j}:")
        #     for pair, count in sorted_pairs[:10]:
        #         print(f"Pair: {pair}, Count: {count}")

        # neg_count, neg_pair = heapq.heappop(pair_pq)
        max_pairs = []
        max_count = max(pair_counts.values())
        for candidate_pair, count in pair_counts.items():
            if count == max_count:
                max_pairs.append(candidate_pair)

        pair = max(max_pairs)
        # if len(max_pairs) > 1:
        #     print(f"Tiebreaking for {pair}")
        #     print(max_pairs)

        # pair = max(pair_counts, key=pair_counts.get)
        # print(pair_counts[pair], pair)

        # count = -neg_count
        # breakpoint()
        # pair = positive_bytes(neg_pair)
        first, second = pair
        new_token = first + second
        pairs_to_update = pair_locations[pair]
        vocab[INITIAL_BYTE_VOCAB_SIZE + len(special_tokens) + j] = new_token
        merges.append(pair)

        # affected_pairs = set()
        # Add the new token and merge pairs
        for word_idx in pairs_to_update:
            pre_merge_word, word = subwords[word_idx], subwords[word_idx]
            word_freq = subword_freqs[tuple(word)]

            # Start from the back because we will pop elements which messes up indices
            for pos in range(len(word) - 1):
                # Since scanning from left to right, we need to continue checking that the indices are correct
                if pos + 1 >= len(word):
                    continue

                # Check that (word[pos], word[pos + 1]) = pair
                if word[pos] != first or word[pos + 1] != second:
                    continue

                # Remove (prev, first)
                if pos > 0:
                    prev_word = word[pos - 1]
                    prev_pair = (prev_word, first)
                    # if bytes('.', encoding="utf-8") in word:
                    #     print(f"{word}, prev pair: {prev_pair}, word_freq: {word_freq}")
                    pair_counts[prev_pair] -= word_freq
                    if pair_counts[prev_pair] == 0:
                        pair_counts.pop(prev_pair)
                    # print(f"Decrementing prev pair {prev_pair} to {pair_counts[prev_pair]} where word freq is {word_freq}")
                    # affected_pairs.add(prev_pair)

                # Remove (second, next)
                if pos < len(word) - 2:
                    next_word = word[pos + 2]
                    next_pair = (second, next_word)
                    pair_counts[next_pair] -= word_freq
                    if pair_counts[next_pair] == 0:
                        pair_counts.pop(next_pair)

                    # if bytes('.', encoding="utf-8") in word:
                    #     print(f"{word}, next pair: {next_pair}")
                    # print(f"Decrementing next pair {next_pair} to {pair_counts[next_pair]} where word freq is {word_freq}")
                    # affected_pairs.add(next_pair)

                # For each word, delete old word, merge token in word
                # Set word frequency to previous
                # NOTE(chris): POP when 0?
                # subword_freqs[tuple(word)] -= word_freq
                word = word[:pos] + (new_token,) + word[pos + 2 :]

                # We need this line because the line before does not change the actual memory in
                # the subwords array. So, the list would not change the actual word.
                # Example:
                # (_, t, h, e), (_, t) merge
                # -> (_t, h, e) is what is desired but we need to set it using the line below
                subwords[word_idx] = word
                # subword_freqs[tuple(word)] = subword_freqs.get(tuple(word), 0) + word_freq

                # Check byte to the left of this pair
                if pos > 0:
                    prev_word = word[pos - 1]
                    new_pair = (prev_word, new_token)
                    pair_counts[new_pair] += word_freq
                    pair_locations[new_pair].append(word_idx)
                    # affected_pairs.add(new_pair)

                # Check byte to the right of this pair
                if pos < len(word) - 1:
                    next_word = word[pos + 1]
                    new_pair = (new_token, next_word)
                    pair_counts[new_pair] += word_freq
                    pair_locations[new_pair].append(word_idx)
                    # affected_pairs.add(new_pair)

            subword_freqs[pre_merge_word] -= word_freq
            subword_freqs[subwords[word_idx]] = subword_freqs.get(subwords[word_idx], 0) + word_freq

        pair_locations.pop(pair)
        pair_counts.pop(pair)

        # for affected_pair in affected_pairs:
        #     if pair_counts[affected_pair] > 0:
        #         heapq.heappush(pair_pq, (-pair_counts[affected_pair], negative_ord_tuple(affected_pair)))

    return vocab, merges
