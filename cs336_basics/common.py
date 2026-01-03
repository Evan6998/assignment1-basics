import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def compile_tokenizers(special_tokens: list[str] | None):
    token_re = re.compile(PAT)
    # Match longer special tokens first so overlapping specials pick the longest match.
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        special_split_re = re.compile("|".join(re.escape(s) for s in sorted_specials))
    else:
        special_split_re = None
    return token_re, special_split_re
