import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(
    train_split_file: str,
    tokenizer_dir: str = "tokenizer",
    tokenizer_file: str = "tokenizer.json",
):
    """Trains tokenizer using byte-pair encoding algorithm."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()  # split into words on whitespace
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=[train_split_file], trainer=trainer)

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, tokenizer_file))  # persist to disk


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split-file", required=True, help="Train split data.")
    args = parser.parse_args(sys.argv[1:])
    train_tokenizer(args.train_split_file)
