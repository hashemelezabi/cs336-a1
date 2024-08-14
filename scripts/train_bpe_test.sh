#! /bin/bash

python -m cs336_basics.bpe \
    --input-path data/bpe_test.txt \
    --vocab-size $1 \
    --special-tokens "<|endoftext|>" \
    --name test 