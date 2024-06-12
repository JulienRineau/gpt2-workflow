import multiprocessing as mp
import os

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
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


local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e7)  # 10M tokens per shard, total of 1000 shards
max_shards = 5


fw = load_dataset("HuggingFaceFW/fineweb-edu", "en", split="train", streaming=True)

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # end of text token


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2 ** 16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                )
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
