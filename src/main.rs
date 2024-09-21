use log::{info};
use env_logger;
use clap::Parser;

/// Command line argument parser
#[derive(Parser, Debug)]
#[command(author = "Dipak Krishnan", version = "1.0", about = "Small efficient transformer in Rust.", long_about = None)]
struct Args {
    /// Path to the training data
    #[arg(long)]
    train_data_path: String,

    /// Path to the tokenizer
    #[arg(long)]
    tokenizer_path: String,
}

fn main() {
    // Parse args
    let args = Args::parse();

    // Initialize logging
    env_logger::init();

    // Define constants
    let batch_size = 32;
    let max_seq_len = 128;

}
