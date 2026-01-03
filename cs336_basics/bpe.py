import multiprocessing as mp
import os
import regex as re
from collections import Counter
from typing import BinaryIO
from cs336_basics.common import compile_tokenizers


def _build_initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = bytes(special_token, "utf8")
    return vocab


def _count_chunk(
    args: tuple[str | os.PathLike[str], int, int, str, str | None],
) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, token_pattern, special_pattern = args
    token_re = re.compile(token_pattern)
    special_split_re = re.compile(special_pattern) if special_pattern else None

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    docs = special_split_re.split(chunk) if special_split_re else [chunk]
    counts: Counter[tuple[bytes, ...]] = Counter()
    for doc in docs:
        for match_ in token_re.finditer(doc):
            # IMPORTANT: BPE operates on UTF-8 *bytes*, not Unicode code points.
            bs = match_.group().encode("utf-8")
            counts[tuple(bytes([b]) for b in bs)] += 1
    print(f"Processed chunk {start}-{end}, found {len(counts)} unique pre-tokens.")
    return counts


def _parallel_count_pre_tokens(
    input_path: str | os.PathLike[str],
    boundaries: list[int],
    token_re: re.Pattern[str],
    special_split_re: re.Pattern[str] | None,
    num_processes: int,
) -> Counter[tuple[bytes, ...]]:
    if len(boundaries) < 2:
        return Counter()

    tasks: list[tuple[str | os.PathLike[str], int, int, str, str | None]] = []
    special_pattern = special_split_re.pattern if special_split_re else None
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, token_re.pattern, special_pattern))

    if len(tasks) == 1 or num_processes == 1:
        print("Counting pre-tokens in single process...")
        return _count_chunk(tasks[0])

    print(f"Counting pre-tokens in parallel with {num_processes} processes...")
    with mp.Pool(processes=min(num_processes, len(tasks))) as pool:
        partial_counts = pool.map(_count_chunk, tasks)

    total_counts: Counter[tuple[bytes, ...]] = Counter()
    for counts in partial_counts:
        total_counts.update(counts)
    return total_counts


def _build_pair_counts(tokens: dict[tuple[bytes, ...], int]) -> Counter[tuple[bytes, bytes]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for token, count in tokens.items():
        if len(token) < 2:
            continue
        for pair in zip(token, token[1:]):
            pair_counts[pair] += count
    return pair_counts


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab = _build_initial_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    token_re, special_split_re = compile_tokenizers(special_tokens)
    num_processes = 6
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)
    print(f"The boundaries for chunking the file are: {boundaries}")

    pre_tokens = _parallel_count_pre_tokens(
        input_path=input_path,
        boundaries=boundaries,
        token_re=token_re,
        special_split_re=special_split_re,
        num_processes=num_processes,
    )

    pair_counts = _build_pair_counts(pre_tokens)
    len_vocab = len(vocab)
    while len_vocab < vocab_size:
        most_common_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))
        merges.append(most_common_pair)
        vocab[len_vocab] = most_common_pair[0] + most_common_pair[1]
        len_vocab += 1

        pair0, pair1 = most_common_pair
        new_pre_token: dict[tuple[bytes, ...], int] = {}
        new_pair_counts: Counter[tuple[bytes, bytes]] = pair_counts.copy()

        for token, count in pre_tokens.items():
            len_token = len(token)
            if len_token < 2:
                new_pre_token[token] = count
                continue

            needs_merge = False
            for i in range(len_token - 1):
                if token[i] == pair0 and token[i + 1] == pair1:
                    needs_merge = True
                    break

            if not needs_merge:
                new_pre_token[token] = count
                continue

            # Remove old pair contributions for this token
            if len_token > 1:
                for pair in zip(token, token[1:]):
                    new_pair_counts[pair] -= count
                    if new_pair_counts[pair] == 0:
                        del new_pair_counts[pair]

            i = 0
            new_token: list[bytes] = []

            while i < len_token:
                current = token[i]
                if i + 1 < len_token and current == pair0 and token[i + 1] == pair1:
                    new_token.append(current + token[i + 1])
                    i += 2
                else:
                    new_token.append(current)
                    i += 1

            new_token_tuple = tuple(new_token)
            new_pre_token[new_token_tuple] = count
            if len(new_token_tuple) > 1:
                for pair in zip(new_token_tuple, new_token_tuple[1:]):
                    new_pair_counts[pair] += count

        pre_tokens = new_pre_token
        pair_counts = new_pair_counts
    
    return vocab, merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[str],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list), "Must represent special token as a list of strings"

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

            # Find the special token in the mini chunk
            matches = [
                mini_chunk.find(bytes(split_special_token, 'utf8'))
                for split_special_token in split_special_tokens
            ]
            best = min(i for i in matches)
            if best != -1:
                chunk_boundaries[bi] = initial_position + best
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
