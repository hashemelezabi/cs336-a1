import os
import regex as re
from collections import defaultdict
import argparse
import pickle, json
import heapq

"""
# This class inverts the comparison ordering
# so that Python min heapq can work as a max priority queue.
"""
class InvertedString(str):
    def __lt__(self, other):
        return str.__gt__(self, other)
    
    def __gt__(self, other):
        return str.__lt__(self, other)
    
    def __le__(self, other):
        return str.__ge__(self, other)
    
    def __ge__(self, other):
        return str.__le__(self, other)
    
    def __eq__(self, other):
        return str.__eq__(self, other)
    
    def __ne__(self, other):
        return str.__ne__(self, other)

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

def merge_tokens(pair, pair_indices, pair_counts, words, counts):
    """
    - pair is the tuple of bytes that is being merged.
    - pair_indices is the indices of words that include the pair
        and thus should be updated.
    - pair_counts maps pair to count. Any pairs overlapping with the merged
        pair should be updated in it.
    - words is the list of words (tuple of bytes), e.g. ('l', 'o', 'w')
        or ('l', 'ow') after the pair ('o', 'w') is merged.
    - counts[i] is the frequency of words[i] in the corpus.
    
    Returns: New where_to_update dictionary.
    """
    def find_pairs(pair, word):
        for i in range(len(word) - 1):
            if pair == (word[i], word[i+1]):
                yield i
    merged = pair[0] + pair[1]
    where_to_update = defaultdict(set)
    for i in pair_indices:
        word = words[i]
        start = 0
        new_word = ()
        # Find occurrences of the pair and merge them, updating
        # pair_counts and where_to_update appropriately.
        for pair_idx in find_pairs(pair, word):
            new_word += word[start:pair_idx] + (word[pair_idx] + word[pair_idx+1],)
            start = pair_idx + 2
            # Decrement count of pairs whose second part was the first part of
            # the merged pair, and increment count of newly formed pair.
            # E.g. if the merged pair is ('s', 't'), then if we look at a word
            # containing ('e', 's', 't'), we decrement count of ('e', 's')
            # and increment count of ('e', 'st').
            if pair_idx > 0:
                pair_counts[(word[pair_idx - 1], pair[0])] -= counts[i]
                pair_counts[(word[pair_idx - 1], merged)] += counts[i]
                # (word[pair_idx - 1], merged) is a new pair, so add it to where_to_update
                where_to_update[(word[pair_idx - 1], merged)].add(i)
            # Similarly handle pairs whose first part was the second part of
            # the merged pair.
            if pair_idx < len(word) - 2:
                pair_counts[(pair[1], word[pair_idx + 2])] -= counts[i]
                pair_counts[(merged, word[pair_idx + 2])] += counts[i]
                where_to_update[(merged, word[pair_idx + 2])].add(i)
        # new_word is ('l', 'ow') instead of ('l', 'o', 'w')
        new_word += word[start:]
        words[i] = new_word
    return where_to_update

def add_to_queue(queue, where_to_update, pair_counts):
    """
    Takes pairs from where_to_update and adds them to queue
    with the appropriate priority, then empties where_to_update.
    """
    for pair, to_update in where_to_update.items():
        count = pair_counts[pair]
        pair_string = (pair[0].decode('utf-8'), pair[1].decode('utf-8'))
        heapq.heappush(queue, (
            -count, InvertedString(pair_string[0]), InvertedString(pair_string[1]), to_update
        ))
    where_to_update.clear()

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
    # pair_counts is a dict[tuple[bytes, bytes], int] and
    # where_to_update is a dict[tuple[bytes, bytes], set[int]]
    pair_counts, where_to_update = count_pairs(words, counts)
    # Take pairs from where_to_update and put in priority queue where priority is
    # the count of the pair, breaking ties using the lexicographically greater pair.
    queue = []
    # Drain where_to_update into queue.
    add_to_queue(queue, where_to_update, pair_counts)
    
    print("Words counts:")
    print(dict(zip(words, counts)))
    
    merges = []
    while len(vocab) < vocab_size:
        print(queue)
        print()
        if not queue:
            print(f"Warning: No more pairs to merge, stopping at vocab size {len(vocab)} instead of {vocab_size}.")
            break
        neg_count, str1, str2, pair_indices = heapq.heappop(queue)
        # Turn back to tuple of bytes; it was stored as strings
        # to ensure lexicographical ordering in the priority queue.
        best_pair = (str1.encode('utf-8'), str2.encode('utf-8')) 
        if -neg_count != pair_counts[best_pair]:
            # The pair count has changed due to a merge that overlaps with this pair,
            # so add back to priority queue with new priority.
            heapq.heappush(queue, (
                -pair_counts[best_pair], str1, str2, pair_indices
            ))
            continue
        
        vocab[token_id] = best_pair[0] + best_pair[1]
        token_id += 1
        merges.append(best_pair)

        # Merge the new pair in every word that should be updated. This
        # updates words, where_to_update, and pair_counts.
        where_to_update = merge_tokens(best_pair, pair_indices, pair_counts, words, counts)
        # Drain where_to_update into queue.
        add_to_queue(queue, where_to_update, pair_counts)
    
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
