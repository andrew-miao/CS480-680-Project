"""
Author: Yanting Miao
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from Model import Transformer
from Optim import TransformerOptim
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import IWSLT

def tokenize_en(text):
    spacy_token = spacy.load('en')
    return [tok.text for tok in spacy_token.tokenizer(text)]

def tokenize_de(text):
    spacy_token = spacy.load('de')
    return [tok.text for tok in spacy_token.tokenizer(text)]

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
            trg_input = trg[:, :-1]
            trg_real = trg[:, 1:]
            translate = model(src, trg_input).permute(0, 2, 1)
            loss = criterion(translate, trg_real)
            total_loss += loss.item()

    return total_loss / len(data)

def training(model, train_data, dev_data, n_epochs, criterion, optimizer, device, path):
    train_loss_list = []
    val_loss_list = []
    model.train()
    step = 1
    print_every = len(train_data)
    min_loss = None
    start = time.time()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for src, trg in train_data:
            optimizer.zero_grad()
            src, trg = src.to(device), trg.to(device)
            # shifted to right, for example, trg = "<s>I love cats</s>", trg_input = "<s>I love cats", trg_real = "I love cats</s>"
            trg_input = trg[:, :-1]
            trg_real = trg[:, 1:]
            translate = model(src, trg_input).permute(0, 2, 1)
            loss = criterion(translate, trg_real)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1
            if step % print_every == 0:
                val_loss = evaluating(model, dev_data, criterion, device)
                m, s = calculate_time(start)
                train_loss_list.append(running_loss / len(train_data))
                val_loss_list.append(val_loss)
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
    return train_loss_list, val_loss_list

if __name__ == '__main__':
    n_epochs = 10
    max_seq = 60
    optim_name = 'Adam'
    print('Loading IWSLT dataset')
    train_data = torch.load('train_loader.pt')
    dev_data = torch.load('dev_loader.pt')
    test_data = torch.load('test_loader.pt')
    src_vocab2num = torch.load('src_vocab2num.pt')
    trg_vocab2num = torch.load('trg_vocab2num.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(len(src_vocab2num), len(trg_vocab2num), 0, 0, 1, 1, device, max_seq=max_seq, d_ff=1024).to(device)
    path = 'best_adam_transformer.pt'
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    adam_optim = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    optimizer = TransformerOptim(adam_optim)
    print('Start training')
    start = time.time()
    train_loss, val_loss = training(model, train_data, dev_data, n_epochs, criterion, optimizer, device, path)
    m, s = calculate_time(start)
    print('Training took %dm%ds' % (m, s))
    print('Start testing')
    model = torch.load(path)
    model = model.to(device)
    test_loss = evaluating(model, test_data, criterion, device)
    print('Test loss: %.3f' % (test_loss))
    print('Saving experiment result')
    train_loss_path = optim_name + '_train_loss.pt'
    val_loss_path = optim_name + '_val_loss.pt'
    test_loss_path = optim_name + '_test_loss.pt'
    torch.save(train_loss, train_loss_path)
    torch.save(val_loss, val_loss_path)
    torch.save(test_loss, test_loss_path)