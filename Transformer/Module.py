"""
Author: Yanting Miao
"""
import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    """
    Create Embedding of input/output.
    """
    def __init__(self, vocab_size, d_model=512):
        """
        :param vocab_size: the number of tokens.
        :param d_model: the dimension of embedding output, default value = 512.
        """
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        :param x: the input.
        :return: embedding pf input.
        """
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    """
    Add positional information to input embeddings.
    """
    def __init__(self, max_seq=5000, d_model=512):
        """
        :param max_seq: the max sequence length, default value = 5000.
        :param d_model: the embedding dimension, default value = 512.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq, d_model)
        position = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(dim=1)
        dim_div = torch.exp((torch.arange(0, d_model, step=2, dtype=torch.float) / d_model) * (-math.log(10000)))
        pe[:, 0:2] = torch.sin(torch.matmul(position, dim_div))
        pe[:, 1:2] = torch.cos(torch.matmul(position, dim_div))
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: the embedding result.
        :return: embedding + position information.
        """
        x.add_(self.pe)
        return x