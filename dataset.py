import multiprocessing as mp
import os

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tqdm import tqdm


class LocalTextDataset(Dataset):
    """Custom Dataset for loading and processing text for language modeling."""

    def __init__(self, sequence_length: int):
        """
        Initializes the LocalTextDataset with the path to the text file and sequence length (block size).

        Args:
            file_path (str): The path to the text file.
            sequence_length (int): The number of tokens in each sample
        """
        with open("input.txt", "r") as f:
            text = f.read()

        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self):
        """Returns the total number of samples available."""
        return len(self.tokens) // self.sequence_length

    def __getitem__(self, idx):
        """Generates one sample of data."""
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length + 1  # +1 for target shift
        buf = self.tokens[start_idx:end_idx]
        x = buf[:-1]  # inputs
        y = buf[1:]  # targets
        return x, y


def tokenize(doc):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    tokens = [eot]  # the special  token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"][0]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

class HuggingFacePreparedTextDataset(Dataset):
    """Custom Dataset for loading and processing large text documents from Hugging Face datasets for language modeling."""
    
    def __init__(self, dataset_name, dataset_config, split, sequence_length: int = 1024, debug_mode: bool = False):
        """
        Initializes the dataset with dataset information and sequence length.
        Args:
        dataset_name (str): Name of the dataset in Hugging Face datasets.
        dataset_config (str): Specific configuration or subset of the dataset.
        split (str): Which split of the dataset to load (e.g., 'train').
        sequence_length (int): The number of tokens in each sample.
        debug_mode (bool): If True, use only the first 100 elements of the dataset.
        """
        self.sequence_length = sequence_length
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens["<|endoftext|>"]
        self.dataset = load_dataset(
            dataset_name, dataset_config, split=split, streaming=True
        )
        
        if debug_mode:
            print("Debug mode: Using only the first 100 elements of the dataset.")
            self.dataset = self.dataset.select(range(min(100, len(dataset))))
        
        print("Calculating document lengths...")
        self.doc_lengths = []
        self.total_samples = 0
        self.cumulative_samples = [0]
        
        for doc in tqdm(self.dataset, desc="Analyzing documents", unit="doc"):
            length = len(self.tokenize(doc))
            self.doc_lengths.append(length)
            samples = max(0, (length - 1) // self.sequence_length)
            self.total_samples += samples
            self.cumulative_samples.append(self.total_samples)
        
        print(f"Total samples: {self.total_samples}")
        if self.total_samples == 0:
            print("Warning: No samples could be created with the given sequence length. Try reducing the sequence length.")

    def tokenize(self, doc):
        """
        Tokenize a single document.
        """
        tokens = [self.eot]  # the special token delimits all documents
        tokens.extend(self.enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return max(1, self.total_samples)  # Return at least 1 to avoid empty dataset errors

    def __getitem__(self, idx):
        """
        Returns a specific sample from the dataset.
        
        Args:
        idx (int): Index of the sample to retrieve.
        
        Returns:
        tuple: A tuple containing input tensor (x) and target tensor (y).
        """
        if self.total_samples == 0:
            # If no valid samples, return a dummy sample
            return torch.zeros(self.sequence_length, dtype=torch.long), torch.zeros(self.sequence_length, dtype=torch.long)

        idx = idx % self.total_samples  # Ensure idx is within range

        # Find which document this index corresponds to
        doc_idx = next(i for i, count in enumerate(self.cumulative_samples) if count > idx) - 1

        # Get the document and tokenize it
        document = self.dataset[doc_idx]
        tokenized_doc = self.tokenize(document)

        # Calculate start and end indices for this sample
        start_idx = (idx - self.cumulative_samples[doc_idx]) * self.sequence_length
        end_idx = start_idx + self.sequence_length + 1  # +1 for target shift

        # Get the chunk of tokens
        buf = tokenized_doc[start_idx:end_idx]

        # Pad if necessary
        if len(buf) < self.sequence_length + 1:
            buf = np.pad(buf, (0, self.sequence_length + 1 - len(buf)), mode='constant', constant_values=self.eot)

        x = buf[:-1]  # inputs
        y = buf[1:]   # targets

        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

    def __iter__(self):
        """
        Returns an iterator over tokenized chunks of sequence_length from the dataset.
        """
        for idx in tqdm(range(len(self)), desc="Iterating over dataset", unit="sample"):
            yield self.__getitem__(idx)

class HuggingFaceStreamingTextDataset(IterableDataset):
    """Custom Iterable Dataset for loading and processing large text documents from Hugging Face datasets for language modeling."""

    def __init__(self, dataset_name, dataset_config, split, sequence_length, nb_tokens):
        """
        Initializes the dataset with dataset information and sequence length.

        Args:
            dataset_name (str): Name of the dataset in Hugging Face datasets.
            dataset_config (str): Specific configuration or subset of the dataset.
            split (str): Which split of the dataset to load (e.g., 'train').
            sequence_length (int): The number of tokens in each sample.
        """
        self.dataset = load_dataset(
            dataset_name, dataset_config, split=split, streaming=True
        )
        self.sequence_length = sequence_length
        self.nb_tokens = nb_tokens

    def __iter__(self):
        """
        Returns an iterator over tokenized chunks of sequence_length from the dataset.
        """
        for document in self.dataset:
            tokenized_doc = tokenize(document)

            num_samples = len(tokenized_doc) // self.sequence_length
            for idx in range(num_samples):
                start_idx = idx * self.sequence_length
                end_idx = start_idx + self.sequence_length + 1  # +1 for target shift
                buf = tokenized_doc[start_idx:end_idx]

                if len(buf) < self.sequence_length + 1:
                    continue

                x = buf[:-1]  # inputs
                y = buf[1:]  # targets

                x = x.astype(np.int64)
                y = y.astype(np.int64)

                yield torch.tensor(x, dtype=torch.long), torch.tensor(
                    y, dtype=torch.long
                )

    def __len__(self):
        """Returns the total number of samples available."""
        return len(self.nb_tokens) // self.sequence_length


if __name__ == "__main__":
    sequence_length = 1024
    dataset = HuggingFaceStreamingTextDataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", sequence_length
    )
    data_iter = iter(dataset)

    dataloader = DataLoader(
        HuggingFaceStreamingTextDataset(
            dataset_name="HuggingFaceFW/fineweb-edu",
            dataset_config="sample-10BT",
            split="train",
            sequence_length=32,
            nb_tokens=9.6e9,  # Because of streaming cannot have the exact number of token so use an estimate
        ),
        batch_size=4,
    )

    for i, data in enumerate(dataloader):
        inputs, targets = data
        print("Batch", i + 1)
        print("Inputs:", inputs)
        print("Targets:", targets)
        if i == 4:
            break
