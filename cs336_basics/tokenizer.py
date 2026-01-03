from typing import Iterator, Self, Iterable
from regex import Pattern
from cs336_basics.bpe import compile_tokenizers

class BPETokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.vocab_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ) -> Self:
        ...

    def _apply_merges(self, token: list[bytes]) -> list[bytes]:
        """Apply learned merges to a single pre-token (list of byte chunks)."""
        for merge in self.merges:
            i = 0
            new_token: list[bytes] = []
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i + 1]) == merge:
                    new_token.append(token[i] + token[i + 1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            token = new_token
        return token

    def _encode_segment(self, text: str, token_re: Pattern[str]) -> list[int]:
        """Encode a piece of text that does not include special tokens."""
        segment_ids: list[int] = []
        for match_ in token_re.finditer(text):
            bs = match_.group().encode("utf-8")
            merged_token = self._apply_merges([bytes([b]) for b in bs])
            segment_ids.extend(self.vocab_to_id[b] for b in merged_token)
        return segment_ids

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []

        token_re, special_split_re = compile_tokenizers(self.special_tokens)
        if not special_split_re:
            return self._encode_segment(text, token_re)

        last_idx = 0
        for match_ in special_split_re.finditer(text):
            token_ids.extend(self._encode_segment(text[last_idx:match_.start()], token_re))
            special_token_bytes = match_.group(0).encode("utf-8")
            token_ids.append(self.vocab_to_id[special_token_bytes])
            last_idx = match_.end()

        if last_idx < len(text):
            token_ids.extend(self._encode_segment(text[last_idx:], token_re))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        decoded_bytes = bytearray()
        for id_ in ids:
            decoded_bytes.extend(self.vocab[id_])
        return decoded_bytes.decode("utf-8", errors="replace")
