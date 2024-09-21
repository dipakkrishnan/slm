use std::fs::File;
use std::io::{BufRead, BufReader};
use tch::{Tensor, Kind, Device};
use tokenizers::Tokenizer;

// Define dataset struct
struct Dataset {
    data: Vec<Tensor>,
    tokenizer: Tokenizer,
}


