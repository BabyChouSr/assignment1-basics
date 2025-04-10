import json
import regex as re
from typing import Iterable, Iterator
from cs336_basics.tokenizer.train import PAT, str_to_byte_list

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None =None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.subword_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merge_rank = {self.merges[i]: i for i in range(len(self.merges))}
        # self.merge_set = set(self.merges)

        if self.special_tokens:
            # Sort by the longest token first because we know that these strings could overlap 
            self.special_tokens.sort(key=lambda x: len(x), reverse=True)
            # We use '(' and ')' to denote that we want to capture the tokens in this group so that
            # when we do re.split, we do not remove these special tokens.
            pattern =  '(' + '|'.join(re.escape(token) for token in self.special_tokens) + ')'
            self.special_tokens_pattern = re.compile(pattern)
        else:
            self.special_tokens_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        
        with open(vocab_filepath, "r") as f_vocab:
            vocab_dict = json.load(f_vocab)
            # breakpoint()
            vocab_dict = {int(k): v.encode('latin-1') for k, v in vocab_dict.items()}
            # breakpoint()
            # print(vocab_dict)
        
        merges = []
        with open(merges_filepath, "r") as f_merges:
            for line in f_merges:
                line = line.strip()
                if not line:
                    continue
                # Parse pattern like (token1,token2)
                match = re.match(r'\((.+?),(.+?)\)', line)
                if match:
                    first_token = match.group(1).encode('latin-1')
                    second_token = match.group(2).encode('latin-1')
                    merges.append((first_token, second_token))

        return Tokenizer(
            vocab=vocab_dict,
            merges=merges,
            special_tokens=special_tokens,
        )
    
    def _apply_merges_to_token_fast(self, token_str: str) -> list[bytes]:
        """
        Efficiently apply BPE merges to a single pre-token (string) using a greedy
        algorithm that updates a dictionary of eligible pairs.
        Returns the final list of tokens (as bytes).
        """
        # Represent the token as a list of one-byte tokens.
        word = [bytes([b]) for b in token_str.encode("utf-8")]
        if len(word) == 0:
            return word

        # Build initial mapping: for each index i in word,
        # if the pair (word[i], word[i+1]) is mergeable (i.e., present in merge_rank),
        # record it in the dict 'pairs' with key = i and value = rank.
        pairs = {}
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair in self.merge_rank:
                pairs[i] = self.merge_rank[pair]

        # Now, repeatedly merge the eligible pair with the lowest rank.
        while pairs:
            # Get the index with the minimum rank.
            i = min(pairs, key=lambda k: pairs[k])
            # Verify that the pair is still valid (could have been affected by previous merges)
            current_pair = (word[i], word[i + 1])
            if current_pair not in self.merge_rank:
                del pairs[i]
                continue

            # Merge the pair at index i.
            new_token = word[i] + word[i + 1]
            word[i] = new_token
            del word[i + 1]

            # Remove outdated pairs: the pair starting at i-1 (if exists) and at i.
            for j in [i - 1, i]:
                if j in pairs:
                    del pairs[j]

            # Update pair at position i-1 if applicable.
            if i - 1 >= 0 and i < len(word):
                pair = (word[i - 1], word[i])
                if pair in self.merge_rank:
                    pairs[i - 1] = self.merge_rank[pair]

            # Update pair at position i (new adjacent pair) if applicable.
            if i < len(word) - 1:
                pair = (word[i], word[i + 1])
                if pair in self.merge_rank:
                    pairs[i] = self.merge_rank[pair]

            # Because the word list shortened by one element, shift keys in pairs dict that occur after i.
            new_pairs = {}
            for pos, rank in pairs.items():
                if pos > i:
                    new_pairs[pos - 1] = rank
                else:
                    new_pairs[pos] = rank
            pairs = new_pairs

        return word

    def _encode_chunk(self, text: str) -> list[int]:
        chunks = []
        if self.special_tokens_pattern:
            chunks = re.split(self.special_tokens_pattern, text)
        else:
            chunks = [text]

        tokens = []

        for chunk in chunks:
            if chunk in self.special_tokens:
                bytestring = chunk.encode(encoding="utf-8")
                tokens.append(self.subword_to_id[bytestring])
            else:
                subword_iter = re.finditer(PAT, chunk)
                for subword_match in subword_iter:
                    subword_str = subword_match.group()
                    subword = str_to_byte_list(subword_str)

                    # while True:
                    #     merge_found = False
                    #     # Need to apply merges in the order that we see the merges. Can't just pick a random applicable merge
                    #     for first, second in self.merges:
                    #         pos = 0
                    #         while pos < len(subword) - 1:
                    #             if pos + 1 >= len(subword):
                    #                 break

                    #             pretoken_first, pretoken_second = subword[pos], subword[pos + 1]
                    #             if pretoken_first == first and pretoken_second == second:
                    #                 merge_found = True
                    #                 # Merge the tokens
                    #                 subword[pos] = first + second

                    #                 # Pop the next token post merge
                    #                 subword.pop(pos + 1)
                    #             else:
                    #                 pos += 1

                    #     if not merge_found:
                    #         break
                    merged_word = self._apply_merges_to_token_fast(subword_str)
                    
                    for subword_bytes in merged_word:
                        # breakpoint()
                        tokens.append(self.subword_to_id[subword_bytes])

        return tokens

    def encode(self, text: str) -> list[int]:
        return self._encode_chunk(text)
                    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        document = ""
        eot_token_len = len("<|endoftext|>")
        for text in iterable:
            if "<|endoftext|>" in text:
                eot_token_idx = text.find("<|endoftext|>")
                document += text[:eot_token_idx + eot_token_len]
                for token in self.encode(document):
                    yield token
                document = text[eot_token_idx + eot_token_len:]
            else:
                document += text
        
        if len(document) > 0:
            for token in self.encode(document):
                yield token

    def decode(self, ids: list[int]) -> str:
        byte_array = []
        for id in ids:
            byte_array.append(self.vocab[id])

        byte_array = b''.join(byte_array)
        return byte_array.decode(encoding="utf-8", errors="replace")       
