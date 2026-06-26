from pathlib import Path

from cs336_basics.bpe import BPE

if __name__ == "__main__":
    p = Path(__file__).resolve().parent.parent / "data" / "TinyStoriesV2-GPT4-train.txt"
    bpe = BPE(
        input_path=p,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    print("TRAIN START")
    bpe.train()
    print("TRAIN DONE :)")
