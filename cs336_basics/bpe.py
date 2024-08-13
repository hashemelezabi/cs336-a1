import os
import regex as re
from collections import defaultdict
import argparse
import pickle, json
import heapq

"""
# The two classes below invert the comparison ordering
# so that Python min heapq can work as a max priority queue.
"""
class InvertedBytes(bytes):
    def __lt__(self, other):
        return self > other
    
    def __gt__(self, other):
        return self < other

class InvertedTuple(tuple):
    def __new__(cls, t):
        return tuple.__new__(cls, (InvertedBytes(t[0]), InvertedBytes(t[1])))
    
    def __lt__(self, other):
        return self > other
    
    def __gt__(self, other):
        return self < other

def count_pairs(words, counts):
    # where_to_update maps a pair of bytes to a set of indices of
    # words that contain this pair.
    pair_counts, where_to_update = defaultdict(int), defaultdict(set)
    for i in range(len(words)):
        word = words[i] # tuple of bytes
        for j in range(len(word) - 1):
            pair = (word[j], word[j+1])
            pair_counts[pair] += counts[i]
            where_to_update[pair].add(i)
    return pair_counts, where_to_update

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
    # We store words in a list (instead of dictionary mapping word to count)
    # because we want to access specific words by index when we update
    # only the words that need updating after a merge using indices in
    # where_to_update. We store counts in a separate list so that
    # counts[i] is the count of words[i]. The only reason we do this is
    # to implement the where_to_update functionality. We do create a word_counts
    # dictionary initially and then use it to populate words and counts.
    word_counts = defaultdict(int)
    for t in pretokens:
        # no need to encode into UTF-8 bytes yet
        word_counts[t] += 1
    words = [] # list of tuples of bytes (words)
    counts = [] # list of counts of words
    for word, count in word_counts.items():
        # word is a tuple of bytes sequences (initially invidivual bytes
        # before any BPE merges), e.g. (b'h', b'e', b'l', b'l', b'o').
        word = tuple([c.encode('utf-8') for c in word])
        words.append(word)
        counts.append(count)

    # Count pairs, and store where_to_update word indices.
    pair_counts, where_to_update = count_pairs(words, counts)
    # Take pairs from where_to_update and put in priority queue where priority is
    # the count of the pair, breaking ties using the lexicographically greater pair.
    queue = []
    for pair, indices in where_to_update.items():
        count = pair_counts[pair]
        # pair_string = pair[0].decode('utf-8') + pair[1].decode('utf-8')
        heapq.heappush(queue, (-count, InvertedTuple(pair), indices))
    where_to_update.clear()
    
    
    num_merges = vocab_size - len(vocab)
    merges = []
    for i in range(num_merges):
        pair_counts = count_pairs(pretoken_freq)
        if not pair_counts:
            print(f"Warning: No more pairs to merge, stopping at vocab size {len(vocab)} instead of {vocab_size}.")
            break
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

def parse_args():
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
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    
    # Save the vocab and merges
    os.makedirs(f"bpe_out/{args.name}", exist_ok=True)
    token_to_id = {}
    for k, v in vocab.items():
        if 128 + len(args.special_tokens) <= k < 256 + len(args.special_tokens):
            # These bytes can't be decoded to utf-8 (meaningless on their own)
            continue
        token_to_id[v.decode('utf-8')] = k
    with open(f"bpe_out/{args.name}/vocab.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(token_to_id, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    with open(f"bpe_out/{args.name}/merges.txt", "w") as f:
        for a, b in merges:
            f.write(f"{a.decode('utf-8')} {b.decode('utf-8')}\n")
