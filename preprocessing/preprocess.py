import os
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(
    train_split_file: str = "dataset_splits/train.txt",
    tokenizer_config_file: str = "tokenizer/tokenizer.json",
):
    """Trains tokenizer using byte-pair encoding algorithm."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()  # split into words on whitespace
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=[train_split_file], trainer=trainer)
    tokenizer.save(tokenizer_config_file)  # persist to disk


def save_to_disk(dataset: Dataset, output_dir: str = "dataset_splits"):
    """Saves dataset splits to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in dataset:
        file_path = os.path.join(output_dir, f"{split}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in dataset[split]["text"]:  # save each split to disk
                f.write(item + "\n")


def main():
    """Main handler to load dataset, save it to disk and train tokenizer."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    save_to_disk(dataset)
    train_tokenizer()


if __name__ == "__main__":
    main()
