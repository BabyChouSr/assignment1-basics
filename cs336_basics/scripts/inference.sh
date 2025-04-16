uv run cs336_basics/inference/decode.py \
--model-path models/tinystories-lr-0.004 \
--vocab-path tokenizers/TinyStoriesV2-GPT4-train-vocab.json \
--merges-path tokenizers/TinyStoriesV2-GPT4-train-merges.txt \
--prompt "Once upon a" \
--max-new-tokens 256 \
--temperature 0.7 \
--top-p 1.0


uv run cs336_basics/inference/decode.py \
--model-path models/owt-lr-0.004 \
--vocab-path tokenizers/owt-train-vocab.json \
--merges-path tokenizers/owt_train-merges.txt \
--prompt "Baseball" \
--max-new-tokens 254 \
--temperature 0.7 \
--top-p 1.0