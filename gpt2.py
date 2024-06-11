# based on Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT/

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import get_preferred_device


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):
    """A multilayer perceptron (MLP) module used within transformer blocks as a feed-forward network.

    Attributes:
        fc (nn.Linear): The first fully connected layer.
        gelu (nn.GELU): Gaussian Error Linear Unit activation layer.
        proj (nn.Linear): The second fully connected layer projecting back to embedding dimension.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    """Causal multihead self-attention implementation.

    Attributes:
        c_attn (nn.Linear): Linear layer to create queries, keys, and values.
        c_proj (nn.Linear): Linear layer to project the output of attention back to the embedding dimension.
        bias (torch.Tensor): Buffer for the causal mask.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd
        )  # projects embedding to bigger space to extract Q, K, V
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, seq length, embedding depth (n_embd

        qkv = self.c_attn(x)
        # Split the combined qkv matrix and reshape it to get individual q, k, v matrices
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """A single transformer block containing a layer normalization, a causal self-attention layer,
    another layer normalization, and an MLP.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The full GPT model comprising an embedding layer, multiple transformer blocks,
    a final layer normalization, and a linear layer for language modeling head.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # stands for hidden
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying between the embedding layers and the pre-softmax linear transformation
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)  # iterate all submodules

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx (torch.Tensor): Input tensor of token indices of shape (B, T).

        Returns:
            torch.Tensor,: Logits of shape (B, T, vocab_size).

        Raises:
            AssertionError: If the sequence length T is greater than the model's block size.
        """
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # final layer norm
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )  # flattening out logits because cross entropy do not accept high dimensions
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        """
        Loads a pretrained GPT model from the Hugging Face transformers library based on the specified model type.

        Args:
            model_type (str): Type of the GPT model to load. Valid options are "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl".

        Returns:
            GPT: An instance of the GPT model with weights loaded from the pretrained model.

        Raises:
            AssertionError: If there's a mismatch in the expected and actual configuration of model parameters.
        """
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        logging.info("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class TextDataset(Dataset):
    """Custom Dataset for loading and processing text for language modeling."""

    def __init__(self, T: int):
        """
        Initializes the TextDataset with the path to the text file and sequence length (block size).

        Args:
            file_path (str): The path to the text file.
            T (int): block_size, t number of tokens in each sample (sequence length).
        """
        with open("input.txt", "r") as f:
            text = f.read()

        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = T

    def __len__(self):
        """Returns the total number of samples available."""
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        """Generates one sample of data."""
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1  # +1 for target shift
        buf = self.tokens[start_idx:end_idx]
        x = buf[:-1]  # inputs
        y = buf[1:]  # targets
        return x, y


class TextDataLoader(DataLoader):
    """Custom DataLoader to handle batching for text data."""

    def __init__(self, B: int, T: int, **kwargs):
        """
        Initializes the TextDataLoader with a dataset, batch size, and additional DataLoader arguments.

        Args:
            dataset (Dataset): The dataset from which to load the data.
            B (int): batch_size, the number of samples per batch.
        """
        self.B = B
        self.T = T
        super(TextDataLoader, self).__init__(TextDataset(T=T), batch_size=B, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    num_return_sequences = 5
    max_length = 30

    device = get_preferred_device()
    torch.manual_seed(1337)
    if device == "cuda":
        torch.cuda.manual_seed(1337)
        try:
            torch.set_float32_matmul_precision("high")  # use tensorcores
        except:
            pass
    elif device == "mps":
        torch.mps.manual_seed(1337)

    if device == "cuda":
        train_data_loader = TextDataLoader(B=16, T=1024)
    else:
        train_data_loader = TextDataLoader(B=4, T=32)

    # model_type = "gpt2"
    # model = GPT.from_pretrained(model_type)
    model = GPT(GPTConfig())
    model.to(device)
    try:  # not supported in torch 2.3 with python 3.12+
        logging.info("Compiling model...")
        model = torch.compile(model)
    except:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(50):
        t0 = time.time()
        data_iter = iter(train_data_loader)
        for _ in range(epoch + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                # Restart the iterator if the number of epochs exceeds the number of batches
                data_iter = iter(train_data_loader)
                x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        if device == "cuda":
            with torch.autocast(device_type=device):
                logits, loss = model(x, y)  # Ensure your model returns logits and loss
        else:
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_data_loader.B * train_data_loader.T) / (t1 - t0)
        print(
            f"Epoch {epoch}, Loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}"
        )

    import sys

    sys.exit(0)

    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model, ")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
    x = tokens.to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
