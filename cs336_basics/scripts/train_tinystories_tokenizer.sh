# Train tokenizer
scalene cs336_basics/tokenizer/train_tokenizer.py --split train --dataset TinyStoriesV2-GPT4 --vocab-size 10000

python cs336_basics/tokenizer/find_longest_token.py --vocab-file results/TinyStoriesV2-GPT4-train-vocab.json