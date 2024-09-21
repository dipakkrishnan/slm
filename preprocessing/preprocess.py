from datasets import load_dataset
import os

# Load the Wikitext-2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Save train, validation, and test splits to separate files
def save_data(split: list[str], file_name: str):
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', file_name)
    with open(file_path, "w") as f:
        for line in split:
            f.write(line + "\n")

# Save the train, validation, and test datasets
save_data(dataset['train']['text'], os.path.join("train_data.txt"))
save_data(dataset['validation']['text'], "validation_data.txt")
save_data(dataset['test']['text'], "test_data.txt")

print("Data saved successfully!")
