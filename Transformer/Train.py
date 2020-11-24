"""
Author: Yanting Miao
"""
import time
import torch
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
            translate = model(src, trg)
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
            translate = model(src, trg)
            loss = criterion(translate)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1
            if step % print_every == 0:
                val_loss = evaluating(model, dev_data, criterion, device)
                m, s = calculate_time(start)
                print('%d/%d, (%d%d), train loss: %.3f, val loss: %.3f' %
                      (epoch + 1, n_epochs, m, s, running_loss / len(train_data), val_loss))
                if min_loss is None or min_loss < val_loss:
                    if min_loss:
                        print('Validation loss decreaseing: %.4f --> %.4f' % (min_loss, val_loss))
                    else:
                        print('Validation loss in first epoch is: %.4f' % (val_loss))
                    min_loss = val_loss
                    torch.save(model, path)
                running_loss = 0.0
                model.train()

if __name__ == '__main__':
    train_data = torch.load('train_loader.pt')
