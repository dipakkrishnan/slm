import os
from datasets import load_dataset, Dataset


def fetch_datasets(output_dir: str = "datasets"):
    """
    Simple utility to fetch wikitext datasets from HF hub.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    for split in dataset:
        file_path = os.path.join(output_dir, f"wiki.{split}.tokens")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in dataset[split]["text"]:  # save each split to disk
                f.write(item + "\n")


if __name__ == "__main__":
    fetch_datasets()
