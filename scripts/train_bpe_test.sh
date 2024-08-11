#! /bin/bash

python -m cs336_basics.bpe \
    --input-path data/bpe_test.txt \
    --vocab-size 300 \
    --special-tokens "<|endoftext|>" \
    --name test 