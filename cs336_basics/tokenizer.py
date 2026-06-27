import pickle
from typing import Iterable, Iterator, Self

import regex as re

from .bpe import BPE


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = set(special_tokens) if special_tokens is not None else set([])
        self.merge_to_index = {(m1, m2): merge_idx for merge_idx, (m1, m2) in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> Self:
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        special_token_positions_dict, special_token_positions_list = self._get_special_token_data(text)
        # Encode text chunks + special tokens
        encoded, text_i = [], 0
        for b, e in sorted(special_token_positions_list):
            chunk = text[text_i:b]
            matches = re.finditer(BPE.PAT, chunk)
            for m in matches:
                pretoken = [bytes([b]) for b in m.group().encode("utf-8")]
                encoded += self._encode_pretoken(pretoken)
            if (b, e) in special_token_positions_dict and special_token_positions_dict[(b, e)] is not None:
                special_token_bytes = special_token_positions_dict[(b, e)].encode("utf-8")
                encoded.append(self.inverse_vocab[special_token_bytes])
            text_i = e
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line_str in iterable:
            encoded_line = self.encode(line_str)
            for token_id in encoded_line:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        decoded = []
        for id in ids:
            decoded.append(self.vocab[id])
        decoded_bytes = b"".join(decoded)
        return decoded_bytes.decode("utf-8", errors="replace")

    def _get_special_token_data(self, text: str) -> tuple[dict[tuple[int, int], str | None], list[tuple[int, int]]]:
        # Localize special tokens: they form text chunk boundaries
        special_token_positions_dict = {}
        special_token_positions_list = []
        if len(self.special_tokens) >= 1:
            special_token_regex = "".join(
                r"{}|".format(re.escape(tok)) for tok in sorted(self.special_tokens, key=lambda s: len(s), reverse=True)
            )[:-1]
            matches = re.finditer(special_token_regex, text)
            for m in matches:
                if m.group() != "":
                    m_pos = (m.start(), m.end())
                    special_token_positions_dict[m_pos] = m.group()
                    special_token_positions_list.append(m_pos)
        # Add sentinel value at the end of text
        special_token_positions_dict[(len(text), len(text))] = None
        special_token_positions_list.append((len(text), len(text)))
        return (special_token_positions_dict, special_token_positions_list)

    def _encode_pretoken(self, pretoken: list[bytes]) -> list[int]:
        encoded = []
        merge_idx_to_apply = self._get_merge_idx_to_apply(pretoken)
        while merge_idx_to_apply is not None:
            m_left, m_right = self.merges[merge_idx_to_apply]
            pretoken = self._apply_merge(pretoken, m_left, m_right)
            merge_idx_to_apply = self._get_merge_idx_to_apply(pretoken)
        for pretoken_bs in pretoken:
            encoded.append(self.inverse_vocab[pretoken_bs])
        return encoded

    def _get_merge_idx_to_apply(self, pretoken: list[bytes]) -> int | None:
        if len(pretoken) <= 1:
            return None
        all_merge_idx = []
        for b_left, b_right in zip(pretoken[:-1], pretoken[1:]):
            if (b_left, b_right) in self.merge_to_index:
                all_merge_idx.append(self.merge_to_index[(b_left, b_right)])
        if all_merge_idx == []:
            return None
        else:
            return min(all_merge_idx)

    def _apply_merge(self, pretoken: list[bytes], m_left: bytes, m_right: bytes) -> list[bytes]:
        if len(pretoken) <= 1:
            return pretoken
        updated_pretoken = []
        pretoken_i = 1
        while pretoken_i < len(pretoken):
            if pretoken[pretoken_i - 1] == m_left and pretoken[pretoken_i] == m_right:
                updated_pretoken.append(m_left + m_right)
                pretoken_i += 2
            else:
                updated_pretoken.append(pretoken[pretoken_i - 1])
                pretoken_i += 1
        if pretoken_i == len(pretoken):
            updated_pretoken.append(pretoken[pretoken_i - 1])
        return updated_pretoken
