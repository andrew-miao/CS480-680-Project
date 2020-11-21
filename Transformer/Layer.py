"""
Author: Yanting Miao
"""
import torch.nn as nn
from Sublayer import MultiHeadAttention
from Sublayer import FeedForward

class EncoderLayer(nn.Module):
    """
    Constructing EncoderLayer
    """
    def __init__(self, n_heads=1, d_model=512, d_ff=2048, dropout=0.1):
        """
        :param n_heads: the number of heads.
        :param d_model: the embedding dimension.
        :param d_ff: the hidden dimension of inner-layers.
        :param dropout: the dropout probability.
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attn = MultiHeadAttention(n_heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, general_mask):
        """
        :param x: the embedding input
        :param general_mask:
        :return: the output of one encoder block.
        """
        attn_output = self.multi_head_attn(x, x, x, general_mask)
        return self.ffn(attn_output)


class DecoderLayer(nn.Module):
    """
    Constructing DecoderLayer
    """
    def __init__(self, n_heads=1, d_model=512, d_ff=2048, dropout=0.1):
        """
        :param n_heads: the number of heads.
        :param d_model: the embedding dimension.
        :param d_ff: the hidden dimension of inner-layers.
        :param dropout: the dropout probability.
        """
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attn = MultiHeadAttention(n_heads, d_model, dropout)
        self.multi_head_attn = MultiHeadAttention(n_heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, decoder_input, encoder_output, general_mask, no_peek_mask):
        """
        :param decoder_input: the embedding input of decoder.
        :param encoder_output: the embedding output of encoder.
        :param general_mask: the mask use can be used in both encoder and decoder,
               to zero attention outputs wherever there is just padding in the input sentences.
        :param no_peek_mask: this mask only used in the decoder, to prevent decoder to peek rest sentence.
        :return: the output of one decoder block.
        """
        attn_output = self.masked_multi_head_attn(decoder_input, decoder_input, decoder_input, no_peek_mask)
        attn_output = self.multi_head_attn(attn_output, encoder_output, encoder_output, general_mask)
        return self.ffn(attn_output)