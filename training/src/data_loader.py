import os
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing


class WikiText2Dataset(Dataset):
    """
    Custom Dataset for the WikiText-2 dataset using a trained BPE tokenizer.
    """

    def __init__(
        self,
        data_dir: str = "datasets",
        tokenizer_path: str = "tokenizer/tokenizer.json",
        split: str = "train",
        seq_length: int = 128,
    ):
        """
        Initializes the dataset by loading the tokenizer and encoding the text data.

        Args:
            data_dir (str): Path to the directory containing the dataset files.
            tokenizer_path (str): Path to the trained BPE tokenizer JSON file.
            split (str): Dataset split to use ('train', 'valid', or 'test').
            seq_length (int): Length of each input sequence.
        """
        self.data_dir = data_dir
        self.split = split
        self.seq_length = seq_length
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.encoded_text = self._encode_text()

    def _load_tokenizer(self, tokenizer_path: str) -> Tokenizer:
        """
        Loads the trained BPE tokenizer from a JSON file.

        Args:
            tokenizer_path (str): Path to the tokenizer JSON file.

        Returns:
            Tokenizer: The loaded BPE tokenizer.
        """
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

        tokenizer = Tokenizer.from_file(tokenizer_path)
        return tokenizer

    def _load_text(self) -> str:
        """
        Loads the raw text data from the dataset file.

        Returns:
            str: The raw text data.
        """
        file_name = f"wiki.{self.split}.tokens"
        file_path = os.path.join(self.data_dir, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return text

    def _encode_text(self) -> List[int]:
        """
        Encodes the entire text data into a list of token IDs using the tokenizer.

        Returns:
            List[int]: The encoded text as a list of token IDs.
        """
        text = self._load_text()
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return (len(self.encoded_text) - 1) // self.seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample of input-target pair.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
        """
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        input_seq = self.encoded_text[start_idx : end_idx - 1]
        target_seq = self.encoded_text[start_idx + 1 : end_idx]

        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        return input_tensor, target_tensor
