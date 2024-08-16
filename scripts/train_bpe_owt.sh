#! /bin/bash

python -m cs336_basics.bpe \
    --input-path data/owt_train.txt \
    --vocab-size 32000 \
    --special-tokens "<|endoftext|>" \
    --name OWT 
