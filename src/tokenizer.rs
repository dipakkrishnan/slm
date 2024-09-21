use tokenizers::tokenizer::{Result, Tokenizer};
use tokenizers::Encoding;

pub fn tokenize_batch(texts: Vec<String>, tokenizer: &Tokenizer) -> Result<Vec<Encoding>> {
    // Use the tokenizer to encode the batch of texts
    let encodings = tokenizer.encode_batch(texts, true)?;
    Ok(encodings)
}

// Helper function to load the tokenizer from a file
pub fn load_tokenizer(file_path: &str) -> Result<Tokenizer> {
    Tokenizer::from_file(file_path)
}
