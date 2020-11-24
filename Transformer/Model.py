"""
Author: Yanting Miao
"""
import torch
import torch.nn as nn
import math
from Layer import EncoderLayer
from Layer import DecoderLayer

class PositionalEncoding(nn.Module):
    """
    Add positional information to input embeddings.
    """
    def __init__(self, max_seq=256, d_model=512):
        """
        :param max_seq: the max sequence length, default value = 256.
        :param d_model: the embedding dimension, default value = 512.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq, d_model)
        position = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(dim=1)
        dim_div = torch.exp((torch.arange(0, d_model, step=2, dtype=torch.float) / d_model) * (-math.log(10000))).unsqueeze(dim=1)
        pe[:, 0:2] = torch.sin(position * dim_div)
        pe[:, 1:2] = torch.cos(position * dim_div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: the embedding result.
        :return: embedding + position information.
        """
        x.add_(self.pe)
        return x

class Encoder(nn.Module):
    """
    Build an Encoder.
    """
    def __init__(self, n_src_vocab, pad_idx, n_layers,
                 max_seq=256, n_heads=1, d_model=512, d_ff=2048, dropout=0.1):
        """
        :param n_src_vocab: the number of vocabulary in the source text.
        :param pad_idx: the idx of padding.
        :param n_layers: the number of encoder blocks.
        :param max_seq: the max sequence length that we will encode in PositionEncoding.
        :param n_heads: the number of heads
        :param d_model: the embedding dimension.
        :param d_ff: the hidden dimension of inner layers in Feed-Forward Networks.
        :param dropout: the dropout probability.
        """
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(n_src_vocab, d_model, padding_idx=pad_idx)
        self.position_encoder = PositionalEncoding(max_seq, d_model)
        self.encoder_blocks = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, src_seq, general_mask):
        """
        :param src_seq: the source sequence.
        :param general_mask: the mask use can be used in both encoder and decoder,
               to zero attention outputs wherever there is just padding in the input sentences.
        :return: output of the encoder.
        """
        embedding_src = self.embedding(src_seq)
        encoder_output = self.position_encoder(embedding_src)
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output, general_mask)

        return encoder_output

class Decoder(nn.Module):
    """
    Build a Decoder.
    """
    def __init__(self, n_tgt_vocab, pad_idx, n_layers,
                 max_seq=256, n_heads=1, d_model=512, d_ff=2048, dropout=0.1):
        """
        :param n_tgt_vocab: the number of vocabulary in the target text.
        :param pad_idx: the idx of padding.
        :param n_layers: the number of encoder blocks.
        :param max_seq: the max sequence length that we will encode in PositionEncoding.
        :param n_heads: the number of heads
        :param d_model: the embedding dimension.
        :param d_ff: the hidden dimension of inner layers in Feed-Forward Networks.
        :param dropout: the dropout probability.
        """
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(n_tgt_vocab, d_model, padding_idx=pad_idx)
        self.position_encoder = PositionalEncoding(max_seq, d_model)
        self.decoder_blocks = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, tgt_seq, no_peek_mask, encoder_output, general_mask):
        """
        :param tgt_seq: the target sequence.
        :param no_peek_mask: this mask only used in the decoder, to prevent decoder to peek rest sentence.
        :param encoder_output: the output of encoder.
        :param general_mask: the mask use can be used in both encoder and decoder,
               to zero attention outputs wherever there is just padding in the input sentences.
        :return: the output of decoder.
        """
        embedding_tgt = self.embedding(tgt_seq)
        decoder_output = self.position_encoder(embedding_tgt)
        for block in self.decoder_blocks:
            decoder_output = block(decoder_output, encoder_output, general_mask, no_peek_mask)

        return decoder_output

class Transformer(nn.Module):
    """
    Build a Transformer.
    """
    def __init__(self, n_src_vocab, n_tgt_vocab, src_pad_idx, tgt_pad_idx, n_encoder_layers, n_decoder_layers, device,
                 n_heads=1, d_model=512, d_ff=2048, max_seq=256, dropout=0.1):
        """
        :param n_src_vocab: the number of tokens in source text.
        :param n_tgt_vocab: the number of tokens in target text
        :param src_pad_idx: the idx of padding in source text.
        :param tgt_pad_idx: the idx of padding in target text.
        :param n_encoder_layers: the number of blocks in encoder.
        :param n_decoder_layers: the number of blocks in decoder.
        :param n_src_vocab: cuda or cpu.
        :param n_heads: the number of heads.
        :param d_model: the embedding dimension.
        :param d_ff: the hidden dimension of inner layers in Feed-Forward Networks.
        :param max_seq: the max sequence length that we will encode in PositionEncoding.
        :param dropout: the dropout probability.
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_src_vocab, src_pad_idx, n_encoder_layers,
                               max_seq=max_seq, n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.decoder = Decoder(n_tgt_vocab, tgt_pad_idx, n_decoder_layers,
                               max_seq=max_seq, n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.fc = nn.Linear(d_model, n_tgt_vocab)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.device = device

    @staticmethod
    def generate_general_mask(seq, pad_idx):
        """
        :param seq: the source/target sequence that we feed to the Transformer.
        :param pad_idx: the idx of padding.
        :return: the mask that can zero out the padding part in the sequence. The size of mask = (1, S, S), where S = max_seq.
        """

        return (seq != pad_idx).unsqueeze(-2)


    def generate_no_peek_mask(self, tgt_seq):
        """
        :param tgt_seq: the target sequence.
        :return: no peek mask.
        """
        seq_len = tgt_seq.size(1)
        mask = torch.triu(torch.ones(1, seq_len, seq_len)).bool()
        return mask.permute(0, 2, 1).to(self.device)

    def forward(self, src_seq, tgt_seq):
        """
        :param src_seq: the source sequence.
        :param tgt_seq: the target sequence.
        :return: the output (probability) of the Transformer.
        """
        src_mask = self.generate_general_mask(src_seq, self.src_pad_idx)
        tgt_mask = self.generate_general_mask(tgt_seq, self.tgt_pad_idx) & self.generate_no_peek_mask(tgt_seq)
        encoder_output = self.encoder(src_seq, src_mask)
        decoder_output = self.decoder(tgt_seq, tgt_mask, encoder_output, src_mask)
        return self.fc(decoder_output)