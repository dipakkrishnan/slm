import torch
from torch.utils.data import Dataset, DataLoader


class WikiDataset(Dataset):

    def __init__(
        self, data_dir: str = "datasets", split: str = "train", seq_length: int = 128
    ):
        """
        Args:
            data_dir (str): Path to the directory containing the dataset.
            split (str): One of 'train', 'valid', or 'test'.
            seq_length (int): The length of each input sequence.
        """
        self.data_dir = data_dir
        self.split = split
        self.seq_length = seq_length

        # Load and preprocess the data
        self.text = self._load_text()
        self.vocab = self._build_vocab()
        self.encoded_text = self._encode_text()

    def _load_text(self) -> str:
        """
        Loads the text data from the dataset file.
        """
        file_name = f"wiki.{self.split}.tokens"
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Optional: Preprocess text (e.g., lowercasing, removing special tokens)
        return text

    def _build_vocab(self) -> Dict[str, int]:
        """
        Builds a vocabulary mapping from tokens to indices.
        """
        # Tokenize the text
        tokens = self.text.split()
        # Build a set of unique tokens
        unique_tokens = set(tokens)
        # Create token to index mapping
        token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
        return token_to_idx

    def _encode_text(self) -> List[int]:
        """
        Encodes the text data into a list of token indices.
        """
        tokens = self.text.split()
        encoded = [self.vocab[token] for token in tokens]
        return encoded

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
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

