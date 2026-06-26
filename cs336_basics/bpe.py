import os
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import BinaryIO, List

import regex as re


class BPE:
    # GPT-2 Pretokenizer Regex
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int = 8
    ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pretokenization_dict: defaultdict[tuple[bytes, ...], int] | None = None
        self.vocab: dict[int, bytes] = {num: bytes([num]) for num in range(256)}
        for tok in special_tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8")
        self.merges: list[tuple[bytes, bytes]] = []
        self.num_processes = num_processes
        self.pretokens_index = None

    def train(self):
        self.pretokenize()
        self._create_pretokens_index()
        num_merges = max(0, self.vocab_size - len(self.vocab))
        for i in range(num_merges):
            self._merge_once()
        return (self.vocab, self.merges)

    def pretokenize(self):
        pretokenizer = BPEPretokenizer(self)
        pretokenizer.pretokenize()

    def _create_pretokens_index(self):
        self.pretokens_index: defaultdict[bytes, set] = defaultdict(set)
        for k in self.pretokenization_dict.keys():
            for b in k:
                self.pretokens_index[b].add(k)

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
        if self.pretokens_index is None:
            items_to_scan = self.pretokenization_dict.items()
        else:
            pretokens_sample = self.pretokens_index[pair_to_merge[0]]
            items_to_scan = [(pretoken, self.pretokenization_dict[pretoken]) for pretoken in pretokens_sample]
        pretokens_to_remove = set()
        for bytes_tuple, count in items_to_scan:
            updated_bytes_list, b2_i, did_merge = [], 1, False
            while b2_i < len(bytes_tuple):
                b1, b2 = bytes_tuple[b2_i - 1], bytes_tuple[b2_i]
                if (b1, b2) == pair_to_merge:
                    updated_bytes_list.append(b1 + b2)
                    did_merge = True
                    b2_i += 2
                else:
                    updated_bytes_list.append(b1)
                    b2_i += 1
            if b2_i == len(bytes_tuple):
                updated_bytes_list.append(bytes_tuple[-1])
            if did_merge:
                old_pretoken = bytes_tuple
                new_pretoken = tuple(updated_bytes_list)
                updated_pretokenization_dict[new_pretoken] += count
                if self.pretokens_index is not None:
                    for b in new_pretoken:
                        self.pretokens_index[b].add(new_pretoken)
                pretokens_to_remove.add(old_pretoken)
        for pretoken in pretokens_to_remove:
            if pretoken in self.pretokenization_dict:
                del self.pretokenization_dict[pretoken]
            for b in pretoken:
                if pretoken in self.pretokens_index[b]:
                    self.pretokens_index[b].remove(pretoken)
        for k, v in updated_pretokenization_dict.items():
            self.pretokenization_dict[k] += v
        self.merges.append(pair_to_merge)
        self.vocab[len(self.vocab)] = pair_to_merge[0] + pair_to_merge[1]

    def __repr__(self):
        return f"BPE: input_path {self.input_path} vocab_size {self.vocab_size} special_tokens {self.special_tokens}"


class BPEPretokenizer:
    def __init__(self, bpe: BPE):
        self.bpe = bpe
        self.num_processes = self.bpe.num_processes

    def pretokenize(self):
        if self.bpe.pretokenization_dict is not None:
            return

        with open(self.bpe.input_path, "rb") as f:
            chunk_boundaries = self.find_chunk_boundaries(
                f, self.num_processes, [s.encode("utf-8", errors="strict") for s in self.bpe.special_tokens]
            )
        chunk_start_ends = list(zip(chunk_boundaries[:-1], chunk_boundaries[1:]))

        processes, q = [], Queue()
        for chunk_start, chunk_end in chunk_start_ends:
            p = Process(target=self.pretokenize_chunk, args=(chunk_start, chunk_end, q))
            p.start()
            processes.append(p)

        self.bpe.pretokenization_dict = defaultdict(int)
        chunk_pretokens_dict = q.get()
        q_read = 1
        while chunk_pretokens_dict is not None:
            for k, v in chunk_pretokens_dict.items():
                self.bpe.pretokenization_dict[k] += v
            if q_read != len(processes):
                chunk_pretokens_dict = q.get()
                q_read += 1
            else:
                chunk_pretokens_dict = None

        for p in processes:
            p.join()

    def pretokenize_chunk(self, chunk_start: int, chunk_end: int, ret_queue: Queue):
        pretokenization_dict = defaultdict(int)
        with open(self.bpe.input_path, "rb") as f:
            f.seek(chunk_start)
            content = f.read(chunk_end - chunk_start).decode("utf-8", errors="ignore")
            special_token_idxs = []
            for tok in self.bpe.special_tokens:
                content_tok_idx = content.find(tok)
                while content_tok_idx != -1:
                    special_token_idxs.append([content_tok_idx, content_tok_idx + len(tok)])
                    content_tok_idx = content.find(tok, content_tok_idx + len(tok))
            special_token_idxs = list(sorted(special_token_idxs))
            special_token_idxs += [(len(content), len(content))]
            content_idx = 0
            for special_tok_start_idx, special_tok_end_idx in special_token_idxs:
                chunk = content[content_idx:special_tok_start_idx]
                matches = re.finditer(self.bpe.PAT, chunk)
                for m in matches:
                    pretoken = m.group()
                    pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                    pretokenization_dict[pretoken_bytes] += 1
                content_idx = special_tok_end_idx
        ret_queue.put(pretokenization_dict)

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_tokens: List[bytes],
    ) -> list[int]:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find any special token in the mini chunk
                for split_special_token in split_special_tokens:
                    found_at = mini_chunk.find(split_special_token)
                    if found_at != -1:
                        break

                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
