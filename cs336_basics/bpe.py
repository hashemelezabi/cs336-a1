import os
import regex as re
from collections import defaultdict
import argparse
import pickle

def count_pairs(vocab):
    pair_counts = defaultdict(int)
    for token, freq in vocab.items():
        # token is a tuple of bytes
        for i in range(len(token) - 1):
            pair_counts[(token[i], token[i+1])] += freq
    return pair_counts

def merge_tokens(pair, vocab):
    def find_pairs(pair, token):
        for i in range(len(token) - 1):
            if pair == (token[i], token[i+1]):
                yield i
    v_out = {}
    for token, freq in vocab.items():
        start = 0
        new_token = ()
        for pair_idx in find_pairs(pair, token):
            new_token += token[start:pair_idx] + (token[pair_idx] + token[pair_idx+1],)
            start = pair_idx + 2
        new_token += token[start:]
        v_out[new_token] = freq
    return v_out

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    test_text: str | None = None,
):
    """
    Returns:
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]]
    """
    # Add special tokens and 256 byte values to vocab
    vocab = {}
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode('utf-8')
    token_id = len(special_tokens)
    for byte in range(256):
        vocab[token_id] = bytes([byte])
        token_id += 1
    
    if test_text:
        text = test_text
    else:
        with open(input_path) as f:
            text = f.read()
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = re.findall(PAT, text)

    # pretoken_freq maps a tuple of bytes to int
    pretoken_freq = defaultdict(int)
    for t in pretokens:
        t_bytes_tup = tuple([c.encode('utf-8') for c in t])
        pretoken_freq[t_bytes_tup] += 1
    
    num_merges = vocab_size - len(vocab)
    merges = []
    for i in range(num_merges):
        pair_counts = count_pairs(pretoken_freq)
        max_freq = max(pair_counts.values())
        most_freq_pairs = [p for p in pair_counts if pair_counts[p] == max_freq]
        best_pair = max(
            most_freq_pairs,
            key=lambda x: (x[0].decode('utf-8'), x[1].decode('utf-8'))
        )
        pretoken_freq = merge_tokens(best_pair, pretoken_freq)
        vocab[token_id] = best_pair[0] + best_pair[1]
        token_id += 1
        merges.append(best_pair)
    
    return vocab, merges




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to the input text file to train BPE on",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Size of the BPE vocab",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="+",
        help="Special tokens to add to the BPE vocab",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of dataset",
    )

    args = parser.parse_args()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    
    # Save the vocab and merges
    os.makedirs(f"bpe_out/{args.name}", exist_ok=True)
    with open(f"bpe_out/{args.name}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"bpe_out/{args.name}/merges.pkl", "wb") as f:
        pickle.dump(merges, f)
