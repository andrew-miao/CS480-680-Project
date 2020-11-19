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

class MultiHeadAttention(nn.Module):
    """
    Compute multi-head attention.
    """
    def __init__(self, n_heads, d_model=512, dropout=0.1):
        """
        :param n_heads: the number of heads.
        :param d_model: the embedding dimension.
        :param dropout: the dropout probability.
        """
        super(MultiHeadAttention, self).__init__()

        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def scaledAttention(self, query, key, value, mask=None):
        """
        :param query: the size of query is (B, T, E), B is the batch size, T is the target sequence length, E is the embedding dimension.
        :param key: the size of key is (B, S, E), B is the batch size, S is the source sequence length, E is the embedding dimension.
        :param value: the size of value is (B, S, E), B is the batch size, S is the source sequence length, E is the embedding dimension.
        :param mask: if mask is not None, the scaledAttention will perform as Masked-Scaled-Attention.
        :return: scaled attention score.
        """
        if mask is not None:
            score = self.softmax((torch.bmm(query, key.permute(0, 2, 1)) + mask) / math.sqrt(self.d_model))
        else:
            score = self.softmax(torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(self.d_model))
        return torch.bmm(score, value)

    def forward(self, query, key, value, mask=None, batch_first=False):
        """
        :param query: the size of query is (T, B, E), T is the target sequence length, B is the batch size, E is the embedding dimension.
        :param key: the size of key is (S, B, E), S is the source sequence length, B is the batch size, E is the embedding dimension.
        :param value: the size of value is (S, B, E), S is the source sequence length, B is the batch size, E is the embedding dimension.
        :param mask: mask, the more detail can see https://arxiv.org/abs/1706.03762
        :param batch_first: if batch_first is True, query size = (B, T, E), key size = (B, S, E), and value size = (B, S, E).
        :return: multi-head attention.
        """
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        if not batch_first:
            attn_score = self.scaledAttention(query.permute(1, 0, 2), key.permute(1, 0, 2), value.permute(1, 0, 2), mask)
            attn_score = attn_score.permute(1, 0, 2)
        else:
            attn_score = self.scaledAttention(query, key, value)
        output = self.fc_out(attn_score)
        return output