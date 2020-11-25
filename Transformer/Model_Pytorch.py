import torch.nn as nn
from torch.nn import Transformer

class TransformerModel(nn.Module):
    def __init__(self, n_src_vocab, n_trg_vocab, d_model=512, n_heads=8, n_encoders=6, n_decoders=6, d_ff=2048, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.src_embedding = nn.Embedding(n_src_vocab, d_model)
        self.trg_embedding = nn.Embedding(n_trg_vocab, d_model)
        self.transformer = Transformer(d_model, n_heads, num_encoder_layers=n_encoders, num_decoder_layers=n_decoders,
                                       dim_feedforward=d_ff, dropout=dropout)
        self.fc = nn.Linear(d_model, n_trg_vocab)

    def forward(self, src_seq, trg_seq):
        src_seq, trg_seq = self.src_embedding(src_seq), self.trg_embedding(trg_seq)
        src_seq, trg_seq = src_seq.permute(1, 0, 2), trg_seq.permute(1, 0, 2)  # size = (B, S), where B is the batch size and S is the max seq.
        output = self.transformer(src_seq, trg_seq)
        return self.fc(output).permute(1, 2, 0)