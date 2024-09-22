import os
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class TokenizerTrainer:

    def __init__(self, train_file: str = "datasets/wiki.train.tokens"):
        self.train_file = train_file
        self.setup_tokenizer()

    def setup_tokenizer(self):
        """
        Setup tokenizer trainer object.
        """
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()  # split into words on whitespace
        self.trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )

    def train(self, tokenizer_output_dir: str = "tokenizer"):
        """
        Trains the byte-pair encoding tokenizer using train data.
        """
        self.tokenizer.train(files=[self.train_file], trainer=self.trainer)
        os.makedirs(tokenizer_output_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(tokenizer_output_dir, "tokenizer.json"))


if __name__ == "__main__":
    trainer = TokenizerTrainer()
    trainer.train()
