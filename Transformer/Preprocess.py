"""
Author: Yanting Miao
"""
import torch
from Tokenize import Token
from torchnlp.datasets import wmt_dataset
from torch.utils.data import TensorDataset, DataLoader

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

def build_train_dev_dataset(data, src_vocab2num, trg_vocab2num, max_seq, test=False):
    src_data = torch.zeros(len(data), max_seq)
    trg_data = torch.zeros(len(data), max_seq)
    trg_label = torch.zeros(len(data), max_seq, len(trg_vocab2num))
    for i in range(len(data)):
        for j in range(len(data['src'][i])):
            word = data['src'][i][j]
            if word in src_vocab2num:
                src_data[i][j] = src_vocab2num[word]
            else:
                src_data[i][j] = src_vocab2num['<unk>']

        for j in range(len(data['trg'][i])):
            word = data['trg'][i][j]
            if word in trg_vocab2num:
                trg_data[i][j] = trg_vocab2num[word]
                if not test:
                    trg_label[i][j][trg_vocab2num[word]] = 1
            else:
                trg_data[i][j] = trg_vocab2num['<unk>']
                if not test:
                    trg_label[i][j][trg_vocab2num['<unk>']] = 1

    if test:
        return src_data, trg_data

    return src_data, trg_data, trg_label


if __name__ == '__main__':
    train_data, dev_data, test_data = wmt_dataset(train=True, dev=True, test=True, train_filename='newstest2009', dev_filename='newstest2013', test_filename='newstest2014')
    for i in range(10, 17):
        if i != 13 and i != 14:
            tmp_dir = 'newstest20' + str(i)
            train_tmp = wmt_dataset(train=True, train_filename=tmp_dir)
            train_data.extend(train_tmp)

    print('Start preprocessing')
    train_data, src_vocab2num, trg_vocab2num, max_seq = stats_data(train_data, 'en', 'de')
    dev_data = stats_data(dev_data, 'en', 'de', False)
    test_data = stats_data(test_data, 'en', 'de', False)
    print('Building dataset')
    train_src_data, train_trg_data, train_trg_label = build_train_dev_dataset(train_data, src_vocab2num, trg_vocab2num, max_seq)
    dev_src_data, dev_trg_data, dev_trg_label = build_train_dev_dataset(dev_data, src_vocab2num, trg_vocab2num, max_seq)
    test_src_data, test_trg_data = build_train_dev_dataset(test_data, src_vocab2num, trg_vocab2num, max_seq, test=True)
    print('Saving results')
    torch.save(train_src_data, 'train_src.pt')
    torch.save(train_trg_data, 'train_trg.pt')
    torch.save(train_trg_label, 'train_trg_label.pt')
    torch.save(dev_src_data, 'dev_src.pt')
    torch.save(dev_trg_data, 'dev_trg.pt')
    torch.save(dev_trg_label, 'dev_trg_label.pt')
    torch.save(test_src_data, 'test_src.pt')
    torch.save(test_trg_data, 'test_trg.pt')
    print('Done preprocessing')