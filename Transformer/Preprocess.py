"""
Author: Yanting Miao
"""
import torch
from Tokenize import Token
import numpy as np
from torchnlp.datasets import wmt_dataset
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
def readLangs(lang1, lang2, train_size=0.8, dev_size=0.1):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    raw_data = []
    for l in lines:
        l = l.split('\t')
        raw_data.append({'en': l[0], 'fr': l[1]})

    raw_data = np.asarray(raw_data)
    permute_idx = np.random.permutation(len(raw_data))
    train_size = int(len(raw_data) * train_size)
    dev_size = int(len(raw_data) * dev_size)
    train_idx = permute_idx[:train_size]
    dev_idx = permute_idx[train_size:dev_size + train_size]
    test_idx = permute_idx[dev_size + train_size:]
    train_data = raw_data[train_idx]
    dev_data = raw_data[dev_idx]
    test_data = raw_data[test_idx]
    return train_data, dev_data, test_data

def stats_sentence(sentence, vocab2num, count, max_seq):
    for i in range(len(sentence)):
        if sentence[i] not in vocab2num:
            vocab2num[sentence[i]] = count
            count += 1
    max_seq = max(len(sentence), max_seq)
    return vocab2num, count, max_seq

def stats_data(data, src_lang, trg_lang, train=True):
    raw_src_data, raw_trg_data = [None] * len(data), [None] * len(data)
    src_tokenize = Token(src_lang)
    trg_tokenize = Token(trg_lang)
    src_vocab2num = {'<pad>': 0, '<unk>': 1}
    trg_vocab2num = {'<pad>': 0, '<unk>': 1}
    src_count, trg_count = 2, 2
    max_src_seq, max_trg_seq = 0, 0
    for i in range(len(data)):
        raw_src_data[i] = src_tokenize.tokenizer(data[i][src_lang])
        raw_trg_data[i] = trg_tokenize.tokenizer(data[i][trg_lang])
        if train:
            src_vocab2num, src_count, max_src_seq = stats_sentence(raw_src_data[i], src_vocab2num, src_count, max_src_seq)
            trg_vocab2num, trg_count, max_trg_seq = stats_sentence(raw_trg_data[i], trg_vocab2num, trg_count, max_trg_seq)

    if not train:
        return {'src': raw_src_data, 'trg': raw_trg_data}

    return {'src': raw_src_data, 'trg': raw_trg_data}, src_vocab2num, trg_vocab2num, max(max_src_seq, max_trg_seq)

def build_train_dev_dataset(data, src_vocab2num, trg_vocab2num, max_seq):
    src_data = torch.zeros(len(data['src']), max_seq)
    trg_data = torch.zeros(len(data['trg']), max_seq)
    for i in range(len(data['src'])):
        for j in range(len(data['src'][i])):
            word = data['src'][i][j]
            if word in src_vocab2num:
                src_data[i][j] = src_vocab2num[word]
            else:
                src_data[i][j] = src_vocab2num['<unk>']

    for i in range(len(data['trg'])):
        for j in range(len(data['trg'][i])):
            word = data['trg'][i][j]
            if word in trg_vocab2num:
                trg_data[i][j] = trg_vocab2num[word]

            else:
                trg_data[i][j] = trg_vocab2num['<unk>']


    return src_data, trg_data


if __name__ == '__main__':
    wmt = False
    if wmt:
        src_lang = 'en'
        trg_lang = 'de'
        train_data, dev_data, test_data = wmt_dataset(train=True, dev=True, test=True, train_filename='newstest2009', dev_filename='newstest2013', test_filename='newstest2014')
        for i in range(10, 11):
            if i != 13 and i != 14:
                tmp_dir = 'newstest20' + str(i)
                train_tmp = wmt_dataset(train=True, train_filename=tmp_dir)
                train_data.extend(train_tmp)
    else:
        src_lang = 'en'
        trg_lang = 'fr'
        train_data, dev_data, test_data = readLangs('eng', 'fra')

    print('Start preprocessing')
    train_data, src_vocab2num, trg_vocab2num, max_seq = stats_data(train_data, src_lang, trg_lang)
    dev_data = stats_data(dev_data, src_lang, trg_lang, False)
    test_data = stats_data(test_data, src_lang, trg_lang, False)
    print('Building dataset')
    train_src_data, train_trg_data = build_train_dev_dataset(train_data, src_vocab2num, trg_vocab2num, max_seq)
    dev_src_data, dev_trg_data = build_train_dev_dataset(dev_data, src_vocab2num, trg_vocab2num, max_seq)
    test_src_data, test_trg_data = build_train_dev_dataset(test_data, src_vocab2num, trg_vocab2num, max_seq)
    train_data = TensorDataset(train_src_data, train_trg_data)
    dev_data = TensorDataset(dev_src_data, dev_trg_data)
    test_data = TensorDataset(test_src_data, test_trg_data)
    print('Building dataloader')
    train_loader = DataLoader(train_data, batch_size=400, pin_memory=True)
    dev_loader = DataLoader(dev_data, batch_size=400, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=400, pin_memory=True)
    print('Saving results')
    torch.save(train_loader, 'train_loader.pt')
    torch.save(dev_loader, 'dev_loader.pt')
    torch.save(test_loader, 'test_loader.pt')
    torch.save(max_seq, 'max_seq.pt')
    torch.save(src_vocab2num, 'src_token2num_pt')
    torch.save(trg_vocab2num, 'trg_token2num_pt')
    print('Done preprocessing')