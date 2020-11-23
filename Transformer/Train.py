import torch

train_src = torch.load('train_src.pt')
print(train_src.size())
max_seq = torch.load('max_seq.pt')
print(max_seq)