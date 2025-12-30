from cs336_basics.bpe import train_bpe
import pickle

if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    with open("bpe_vocab.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)
