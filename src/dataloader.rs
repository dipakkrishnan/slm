use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub struct DataLoader {
    file_path: String,
}

impl DataLoader {
    // Constructor to initialize the DataLoader with a file path
    pub fn new(file_path: String) -> Self {
        DataLoader { file_path }
    }

    // Method to load data from the file and return it as a vector of strings
    pub fn load_data(&self) -> Result<Vec<String>, io::Error> {
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);

        let mut lines = Vec::new();
        for line in reader.lines() {
            lines.push(line?);
        }
        Ok(lines)
    }

    // Method to return batches of data with a given batch size
    pub fn get_batches(&self, batch_size: usize) -> Result<Vec<Vec<String>>, io::Error> {
        let data = self.load_data()?;
        let mut batches = Vec::new();

        for chunk in data.chunks(batch_size) {
            batches.push(chunk.to_vec());
        }

        Ok(batches)
    }
}
