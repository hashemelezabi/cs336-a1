#! /bin/bash

python -m cs336_basics.bpe.train \
    --input-path data/bpe_test.txt \
    --vocab-size ${1:-300} \
    --special-tokens "<|endoftext|>" \
    --name test 