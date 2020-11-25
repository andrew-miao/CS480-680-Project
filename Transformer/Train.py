"""
Author: Yanting Miao
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from Model import Transformer
from Optim import TransformerOptim

def calculate_time(start):
    end = time.time()
    t = end - start
    m = t // 60
    s = t - m * 60
    return m, s

def evaluating(model, data, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, trg in data:
            src, trg = src.to(device), trg.to(device)
            translate = model(src, trg).permute(0, 2, 1)
            loss = criterion(translate, trg)
            total_loss += loss.item()

    return total_loss / len(data)

def training(model, train_data, dev_data, n_epochs, criterion, optimizer, device, path):
    model.train()
    step = 1
    print_every = len(train_data)
    min_loss = None
    start = time.time()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for src, trg in train_data:
            src, trg = src.to(device), trg.to(device)
            translate = model(src, trg).permute(0, 2, 1)
            loss = criterion(translate, trg)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1
            if step % print_every == 0:
                val_loss = evaluating(model, dev_data, criterion, device)
                m, s = calculate_time(start)
                print('%d/%d, (%dm%ds), train loss: %.3f, val loss: %.3f' %
                      (epoch + 1, n_epochs, m, s, running_loss / len(train_data), val_loss))
                if min_loss is None or min_loss > val_loss:
                    if min_loss:
                        print('Validation loss decreaseing: %.4f --> %.4f' % (min_loss, val_loss))
                    else:
                        print('Validation loss in first epoch is: %.4f' % (val_loss))
                    min_loss = val_loss
                    torch.save(model, path)
                running_loss = 0.0
                model.train()

if __name__ == '__main__':
    n_epochs = 10
    train_data = torch.load('train_loader.pt')
    dev_data = torch.load('dev_loader.pt')
    test_data = torch.load('test_loader.pt')
    src_token2num = torch.load('src_token2num.pt')
    trg_token2num = torch.load('trg_token2num.pt')
    max_seq = torch.load('max_seq.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(len(src_token2num), len(trg_token2num), 0, 0, 1, 1, device, max_seq=max_seq, d_ff=1024).to(device)
    path = 'best_transformer.pt'
    criterion = nn.CrossEntropyLoss()
    adam_optim = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    optimizer = TransformerOptim(adam_optim)
    print('Start training')
    start = time.time()
    training(model, train_data, dev_data, n_epochs, criterion, optimizer, device, path)
    m, s = calculate_time(start)
    print('Training took %dm%ds' % (m, s))
    print('Start testing')
    model = torch.load(path)
    model = model.to(device)
    test_loss = evaluating(model, test_data, criterion, device)
    print('Test loss: %.3f' % (test_loss))