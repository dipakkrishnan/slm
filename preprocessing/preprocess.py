from datasets import load_dataset

# Load the Wikitext-2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Save train, validation, and test splits to separate files
def save_data(split, file_name):
    with open(file_name, "w") as f:
        for line in split:
            f.write(line + "\n")

# Save the train, validation, and test datasets
save_data(dataset['train']['text'], "train_data.txt")
save_data(dataset['validation']['text'], "validation_data.txt")
save_data(dataset['test']['text'], "test_data.txt")

print("Data saved successfully!")
