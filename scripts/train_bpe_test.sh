#! /bin/bash

python -m cs336_basics.bpe \
    --input-path data/bpe_test.txt \
    --vocab-size 260 \
    --special-tokens "<|endoftext|>" \
    --name test 