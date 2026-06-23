import os
from collections import defaultdict

import regex as re


class BPE:
    # GPT-2 Pretokenizer Regex
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pretokenization_dict: defaultdict[tuple[bytes, ...], int] | None = None
        self.vocab: dict[int, bytes] = {num: bytes([num]) for num in range(256)}
        for tok in special_tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8")
        self.merges: list[tuple[bytes, bytes]] = []

    def train(self):
        self.pretokenize()
        num_merges = max(0, self.vocab_size - len(self.vocab))
        for i in range(num_merges):
            self._merge_once()
        return (self.vocab, self.merges)

    def pretokenize(self):
        if self.pretokenization_dict is not None:
            return
        self.pretokenization_dict = defaultdict(int)
        with open(self.input_path, "rb") as f:
            content = f.read().decode("utf-8", errors="ignore")
            special_token_idxs = []
            for tok in self.special_tokens:
                content_tok_idx = content.find(tok)
                while content_tok_idx != -1:
                    special_token_idxs.append([content_tok_idx, content_tok_idx + len(tok)])
                    content_tok_idx = content.find(tok, content_tok_idx + len(tok))
            special_token_idxs = list(sorted(special_token_idxs))
            special_token_idxs += [(len(content), len(content))]
            content_idx = 0
            for special_tok_start_idx, special_tok_end_idx in special_token_idxs:
                chunk = content[content_idx:special_tok_start_idx]
                matches = re.finditer(self.PAT, chunk)
                for m in matches:
                    pretoken = m.group()
                    pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                    self.pretokenization_dict[pretoken_bytes] += 1
                content_idx = special_tok_end_idx

    def _merge_once(self):
        pairs_dict = defaultdict(int)
        for bytes_tuple in self.pretokenization_dict:
            if len(bytes_tuple) <= 1:
                continue
            for b1, b2 in zip(bytes_tuple[:-1], bytes_tuple[1:]):
                pairs_dict[(b1, b2)] += self.pretokenization_dict[bytes_tuple]
        if len(pairs_dict) == 0:
            return
        pair_to_merge = sorted(pairs_dict.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]
        updated_pretokenization_dict = defaultdict(int)
        for bytes_tuple, count in self.pretokenization_dict.items():
            updated_bytes_list, b2_i = [], 1
            while b2_i < len(bytes_tuple):
                b1, b2 = bytes_tuple[b2_i - 1], bytes_tuple[b2_i]
                if (b1, b2) == pair_to_merge:
                    updated_bytes_list.append(b1 + b2)
                    b2_i += 2
                else:
                    updated_bytes_list.append(b1)
                    b2_i += 1
            if b2_i == len(bytes_tuple):
                updated_bytes_list.append(bytes_tuple[-1])
            updated_pretokenization_dict[tuple(updated_bytes_list)] += count
        self.pretokenization_dict = updated_pretokenization_dict
        self.merges.append(pair_to_merge)
        self.vocab[len(self.vocab)] = pair_to_merge[0] + pair_to_merge[1]

    def __repr__(self):
        return f"BPE: input_path {self.input_path} vocab_size {self.vocab_size} special_tokens {self.special_tokens}"
