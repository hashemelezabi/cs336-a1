import os
from cs336_basics.bpe.tokenizer import Tokenizer
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe_out', help='Path to the BPE output folder')
    parser.add_argument('--dataset', help='Path to the dataset file')
    args = parser.parse_args()

    tokenizer = Tokenizer.from_folder(args.bpe_out)
    with open(args.dataset) as f:
        token_ids = tokenizer.encode(f.read())
    os.makedirs('./tokenize_out', exist_ok=True)
    file_name = os.path.splitext(os.path.basename(args.dataset))[0]
    np.save(f"./tokenize_out/{file_name}_ids.npy", np.array(token_ids, dtype=np.uint16))