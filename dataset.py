import multiprocessing as mp
import os

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tqdm import tqdm


class TextDataset(Dataset):
    """Custom Dataset for loading and processing text for language modeling."""

    def __init__(self, sequence_length: int):
        """
        Initializes the TextDataset with the path to the text file and sequence length (block size).

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
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


class HuggingFaceTextDataset(IterableDataset):
    """Custom Iterable Dataset for loading and processing large text documents from Hugging Face datasets for language modeling."""

    def __init__(self, dataset_name, dataset_config, split, sequence_length):
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


if __name__ == "__main__":
    sequence_length = 1024
    dataset = HuggingFaceTextDataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", sequence_length
    )
    data_iter = iter(dataset)

    dataloader = DataLoader(
        HuggingFaceTextDataset(
            dataset_name="HuggingFaceFW/fineweb-edu",
            dataset_config="sample-10BT",
            split="train",
            sequence_length=32,
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
