"""
Author: Yanting Miao
"""
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Compute multi-head attention.
    """
    def __init__(self, n_heads=1, d_model=512, dropout=0.1):
        # Because of the constrain of hardware (I only have one GPU: RTX 2060 Super), and thus, the default value of n_heads = 1.
        """
        :param n_heads: the number of heads.
        :param d_model: the embedding dimension.
        :param dropout: the dropout probability.
        """
        super(MultiHeadAttention, self).__init__()

        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model, 1e-6)

    def scaledAttention(self, query, key, value, mask=None):
        """
        :param query: the size of query is (B, T, E), B is the batch size, T is the target sequence length, E is the embedding dimension.
        :param key: the size of key is (B, S, E), B is the batch size, S is the source sequence length, E is the embedding dimension.
        :param value: the size of value is (B, S, E), B is the batch size, S is the source sequence length, E is the embedding dimension.
        :param mask: if mask is not None, the scaledAttention will perform as Masked-Scaled-Attention.
        :return: scaled attention score.
        """
        score = self.softmax(torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(self.d_model))
        if mask is not None:
            score.masked_fill_(mask == 0, -1e9)  # use -1e9 to represent -inf in the original paper.
        score = self.dropout(score)
        return torch.bmm(score, value)

    def forward(self, query, key, value, mask=None, batch_first=True):
        """
        :param query: the size of query is (T, B, E), T is the target sequence length, B is the batch size, E is the embedding dimension.
        :param key: the size of key is (S, B, E), S is the source sequence length, B is the batch size, E is the embedding dimension.
        :param value: the size of value is (S, B, E), S is the source sequence length, B is the batch size, E is the embedding dimension.
        :param mask: mask, the more detail can see https://arxiv.org/abs/1706.03762
        :param batch_first: if batch_first is True, query size = (B, T, E), key size = (B, S, E), and value size = (B, S, E).
        :return: multi-head attention.
        """
        residual = query
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        if not batch_first:
            output = self.scaledAttention(query.permute(1, 0, 2), key.permute(1, 0, 2), value.permute(1, 0, 2), mask)
            output = output.permute(1, 0, 2)
        else:
            output = self.scaledAttention(query, key, value)

        output = self.dropout(self.fc_out(output))
        output.add_(residual)
        return self.layernorm(output)

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks in original paper.
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        """
        :param d_model: embedding dimension, default value = 512.
        :param d_ff: the dimension of inner-layer, default value = 2048.
        :param dropout: the probability of dropout.
        """
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, 1e-6)

    def forward(self, x):
        """
        :param x: input of multi-head attention module.
        :return: output of position-wise feed-forward networks module.
        """
        residual = x
        x = self.fc2(self.relu(self.fc1(x)))
        output = self.dropout(x)
        output.add_(residual)
        return self.layernorm(output)