import os
import torch
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


def main():
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Load tokenizer and cast to transformers type
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "[PAD]"

    def tokenize_function(examples: dict):
        return fast_tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_valid.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"])

    os.makedirs("tensors", exist_ok=True)
    torch.save(
        (tokenized_train["input_ids"], tokenized_train["attention_mask"]),
        "tensors/train_data.pt",
    )
    torch.save(
        (tokenized_valid["input_ids"], tokenized_valid["attention_mask"]),
        "tensors/valid_data.pt",
    )
    torch.save(
        (tokenized_test["input_ids"], tokenized_test["attention_mask"]),
        "tensors/test_data.pt",
    )


if __name__ == "__main__":
    main()
