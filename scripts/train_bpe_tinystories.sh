#! /bin/bash

python -m cs336_basics.bpe.train \
    --input-path data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 \
    --special-tokens "<|endoftext|>" \
    --name TinyStories 