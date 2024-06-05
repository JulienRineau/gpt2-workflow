import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from utils import get_preferred_device


class TransformerModel(nn.Module):
    """
    A Transformer Model for sequence modeling.

    Attributes:
        model_type (str): Type of the model, set to 'Transformer'.
        pos_encoder (PositionalEncoding): Positional Encoding module.
        transformer_encoder (TransformerEncoder): Transformer Encoder module.
        embedding (nn.Embedding): Embedding layer for input tokens.
        d_model (int): Dimensionality of the model.
        linear (nn.Linear): Linear layer to map from hidden state to token space.

    Parameters:
        ntoken (int): Number of tokens (vocabulary size).
        d_model (int): Dimensionality of the model (embeddings size).
        nhead (int): Number of heads in the multi-head attention models.
        d_hid (int): Dimensionality of the feedforward network model in nn.TransformerEncoder.
        nlayers (int): Number of nn.TransformerEncoderLayer in nn.TransformerEncoder.
        dropout (float, optional): Dropout value (default=0.5).
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.device = get_preferred_device()

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes weights of the Transformer model."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Pass the input through the Transformer model.

        Args:
            src (Tensor): Input tensor of shape [seq_len, batch_size].
            src_mask (Tensor, optional): Mask for src tensor of shape [seq_len, seq_len].

        Returns:
            Tensor: Output tensor of shape [seq_len, batch_size, ntoken].
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(
                self.device
            )
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings using sine and cosine functions.

    Attributes:
        dropout (nn.Dropout): Dropout module to apply after adding positional encoding.

    Parameters:
        d_model (int): The dimensionality of the embeddings.
        dropout (float): Dropout rate.
        max_len (int): The maximum length of the sequence to be encoded.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply positional encoding to the input tensor.

        Args:
            x (Tensor): The input tensor of shape [seq_len, batch_size, embedding_dim].

        Returns:
            Tensor: The encoded tensor with the same shape as `x`.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
