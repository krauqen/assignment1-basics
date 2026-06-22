import os


class BPE:
    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def __repr__(self):
        return f"BPE: input_path {self.input_path} vocab_size {self.vocab_size} special_tokens {self.special_tokens}"
