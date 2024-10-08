import pickle
import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.id2token = vocab
        self.merge2idx = {merge: idx for idx, merge in enumerate(merges)}
        self.token2id: dict[bytes, int] = {token: token_id for token_id, token in vocab.items()}
        self.special_tokens = special_tokens

        # add special tokens if they aren't already in vocab
        if special_tokens:
            for t in special_tokens:
                tok_enc = t.encode('utf-8')
                if tok_enc not in self.token2id:
                    self.id2token[len(vocab)] = tok_enc
                    self.token2id[tok_enc] = len(vocab) - 1
        
        # compile regex pattern for pretokenization
        self.pretokenize_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    @classmethod
    def from_folder(cls, folder_path, special_tokens=None):
        return cls.from_files(f"{folder_path}/vocab.pkl", f"{folder_path}/merges.pkl", special_tokens)
    
    def _find_pairs(pair: tuple[bytes, bytes], word: tuple[bytes, ...]):
        for i in range(len(word) - 1):
            if pair == (word[i], word[i+1]):
                yield i

    def _apply_merge(self, pair: tuple[bytes, bytes], token: tuple[bytes, ...]):
        def find_pairs(pair, token):
            for i in range(len(token) - 1):
                if pair == (token[i], token[i+1]):
                    yield i
        new_token = ()
        start = 0
        for pair_idx in find_pairs(pair, token):
            new_token += token[start:pair_idx] + (token[pair_idx] + token[pair_idx+1],)
            start = pair_idx + 2
        new_token += token[start:]
        return new_token
    
    def _find_all(self, string, sub):
        # Adapted from https://stackoverflow.com/a/4665027/5109078
        slices = []
        start = 0
        while True:
            start = string.find(sub, start)
            if start == -1:
                break
            slices.append((start, start + len(sub)))
            start += len(sub)
        return slices
    
    def _get_pretokens(self, text):
        """
        Returns iterator over pretokens.
        """
        # Avoid splitting special tokens by finding them
        # and adding them intact to `pretokens`.
        special_token_slices = []
        if self.special_tokens:
            for special_tok in self.special_tokens:
                special_token_slices.extend(self._find_all(text, special_tok))
        if not special_token_slices:
            for match in self.pretokenize_pat.finditer(text):
                yield text[match.start():match.end()]
        else:
            special_token_slices.sort()
             # Remove slices contained in other slices, e.g.
             # for "<|endoftext|><|endoftext|>", should remove
             # the slice for a single "<|endoftext|>".
            slices_filtered = []
            prev_end = -1
            for i in range(len(special_token_slices)):
                slice = special_token_slices[i]
                if i < len(special_token_slices) - 1 and special_token_slices[i+1][0] == slice[0]:
                    # Next slice has same start as this slice but is bigger (has bigger end),
                    # since we sorted.
                    continue
                if i > 0 and slice[1] <= prev_end:
                    # If previous slice was (11, 37) and this slice is (24, 37), skip this slice
                    # since it's contained in previous one.
                    continue
                slices_filtered.append(slice)
                prev_end = slice[1]
            # Now pretokenize the parts around the special tokens
            # and yield the special tokens intact.
            for i in range(len(slices_filtered)):
                start = 0 if i == 0 else slices_filtered[i-1][1]
                for match in self.pretokenize_pat.finditer(text[start:slices_filtered[i][0]]):
                    yield text[start + match.start():start + match.end()]
                slice_start, slice_end = slices_filtered[i]
                # Yield the special token as is.
                yield text[slice_start:slice_end]
            # Yield the remaining non-special token pretokens.
            start = slices_filtered[-1][1]
            for match in self.pretokenize_pat.finditer(text[start:]):
                yield text[start + match.start():start + match.end()]

    def encode(self, text: str) -> list[int]:
        token_ids = []
        for pretoken in self._get_pretokens(text):
            pretoken_enc = pretoken.encode('utf-8')
            if pretoken_enc in self.token2id:
                # This handles special tokens. It also immediately
                # gets the token id if the entire token is in the vocab
                # instead of progressively applying BPE merges until
                # we get the full token. Not sure why the algorithm
                # from the assignment doesn't mention this case.
                token_ids.append(self.token2id[pretoken_enc])
                continue
            # Apply BPE merges until we can't apply them anymore.
            token = tuple([bytes([b]) for b in pretoken.encode('utf-8')])
            while len(token) >= 2:
                pairs_in_token = set(zip(token, token[1:]))
                earliest_merge = min(pairs_in_token, key=lambda p: self.merge2idx.get(p, float('inf')))
                if earliest_merge not in self.merge2idx:
                    # No more applicable merges.
                    break
                token = self._apply_merge(earliest_merge, token)
            token_ids.extend([self.token2id[subword] for subword in token])
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for pretoken in self._get_pretokens(text):
                pretoken_enc = pretoken.encode('utf-8')
                if pretoken_enc in self.token2id:
                    yield self.token2id[pretoken_enc]
                else:
                    # Apply BPE merges until we can't apply them anymore.
                    token = tuple([bytes([b]) for b in pretoken.encode('utf-8')])
                    # https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py#L57
                    while len(token) >= 2:
                        pairs_in_token = set(zip(token, token[1:]))
                        earliest_merge = min(pairs_in_token, key=lambda p: self.merge2idx.get(p, float('inf')))
                        if earliest_merge not in self.merge2idx:
                            # No more applicable merges.
                            break
                        token = self._apply_merge(earliest_merge, token)
                    # Yield IDs.
                    for subword in token:
                        yield self.token2id[subword]
    
    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b''
        for token_id in ids:
            if token_id not in self.id2token:
                raise ValueError(f"Token ID {token_id} not in vocabulary.")
            decoded_bytes += self.id2token[token_id]
        return decoded_bytes.decode('utf-8', errors='replace')