mod dataloader;
mod tokenizer;

use dataloader::DataLoader;
use tokenizer::{load_tokenizer, tokenize_batch};
use log::{info};
use env_logger;

fn main() {
    // Initialize logging
    env_logger::init();

    // Create a DataLoader instance to load data from train.txt
    let raw_train_data_path = "dataset_splits/train.txt";
    let data_loader = DataLoader::new(raw_train_data_path.to_string());
    info!("Loaded training data from {}", raw_train_data_path);

    let tokenizer_path = "tokenizer/tokenizer.json";
    // Load the tokenizer from tokenizer.json config
    let tokenizer = load_tokenizer(tokenizer_path).unwrap();
    info!("Loaded tokenizer from {}", tokenizer_path);

    // Define the batch size (e.g., 32 texts per batch)
    let batch_size = 32;

    // Load the dataset and split it into batches
    match data_loader.get_batches(batch_size) {
        Ok(batches) => {
            for batch in batches {
                match tokenize_batch(batch, &tokenizer) {
                    Ok(encodings) => {
                        // Process and display tokenized results
                        for encoding in encodings {
                            println!("{:?}", encoding.get_tokens());
                        }
                    }
                    Err(e) => {
                        eprintln!("Error tokenizing batch: {:?}", e);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error loading dataset: {:?}", e);
        }
    }
}
