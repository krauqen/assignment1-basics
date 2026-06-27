from pathlib import Path

from cs336_basics.bpe import BPE

if __name__ == "__main__":
    path_in = Path(__file__).resolve().parent.parent / "data" / "TinyStoriesV2-GPT4-train.txt"
    vocab_path_out = Path(__file__).resolve().parent.parent / "data" / "TinyStoriesV2-GPT4-train-BPE-vocab.pkl"
    merges_path_out = Path(__file__).resolve().parent.parent / "data" / "TinyStoriesV2-GPT4-train-BPE-merges.pkl"
    bpe = BPE(
        input_path=path_in,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    bpe.train()
    bpe.save(vocab_path_out, merges_path_out)
