from cs336_basics.tokenizer.train import train_bpe

if __name__ == "__main__":
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 100, ["<|endoftext|>"])