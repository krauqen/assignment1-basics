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
        self.special_tokens_bytes = (
            set([tok.encode("utf-8") for tok in special_tokens]) if special_tokens is not None else set([])
        )

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
                for b1, b2 in self.merges:
                    if len(pretoken) <= 1:
                        break
                    updated_pretoken = []
                    pretoken_i = 1
                    while pretoken_i < len(pretoken):
                        if pretoken[pretoken_i - 1] == b1 and pretoken[pretoken_i] == b2:
                            updated_pretoken.append(b1 + b2)
                            pretoken_i += 2
                        else:
                            updated_pretoken.append(pretoken[pretoken_i - 1])
                            pretoken_i += 1
                    if pretoken_i == len(pretoken):
                        updated_pretoken.append(pretoken[pretoken_i - 1])
                    pretoken = updated_pretoken
                for pretoken_bs in pretoken:
                    encoded.append(self.inverse_vocab[pretoken_bs])
            if (b, e) in special_token_positions_dict and special_token_positions_dict[(b, e)] is not None:
                special_token_bytes = special_token_positions_dict[(b, e)].encode("utf-8")
                encoded.append(self.inverse_vocab[special_token_bytes])
            text_i = e
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # chunk_size = 4096
        # buffer_size = 1024
        # content = iterable.read(chunk_size + buffer_size)
        # while content != "":
        #     special_token_positions_dict, special_token_positions_list = self._get_special_token_data(content)
        #     chunk =
        #     content += iterable.read(chunk_size + buffer_size)
        raise NotImplementedError

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
        if len(self.special_tokens) >= 0:
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
