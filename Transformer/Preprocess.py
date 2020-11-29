import spacy
from torchtext.data import Field
from torchtext.datasets import IWSLT
import torch
from torch.utils.data import TensorDataset, DataLoader
from Tokenize import Token
import numpy as np

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def buildTensor(dataset, max_seq, src_vocab2num, trg_vocab2num, train=True, dev=False, test=False):
    src_data = torch.ones(len(dataset), max_seq + 2)
    trg_data = None
    if not train:
        raw_trg = []
    if not test:
        trg_data = torch.ones(len(dataset), max_seq + 2)

    for i, sentence in enumerate(dataset):
        src_data[i][0] = src_vocab2num[BOS]
        src_data[i][-1] = trg_vocab2num[EOS]
        for j in range(1, min(max_seq + 1, len(sentence.src))):
            word = sentence.src[j - 1]
            src_data[i][j] = src_vocab2num[word]

        if trg_data is not None:
            trg_data[i][0] = trg_vocab2num[BOS]
            trg_data[i][-1] = trg_vocab2num[EOS]
            for j in range(1, min(max_seq + 1, len(sentence.trg))):
                word = sentence.trg[j - 1]
                trg_data[i][j] = trg_vocab2num[word]

            if not train:
                raw_trg.append(sentence.trg)

    if trg_data is not None and train:
        return src_data, trg_data

    elif trg_data is not None and dev:
        return src_data, trg_data, raw_trg

    else:
        return src_data, raw_trg

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
    src_vocab2num = {'<unk>': 0, '<pad>': 1, '<s>': 2, '</s>': 3}
    trg_vocab2num = {'<unk>': 0, '<pad>': 1, '<s>': 2, '</s>': 3}
    src_count, trg_count = 4, 4
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
    src_data = torch.ones(len(data['src']), max_seq)
    trg_data = torch.ones(len(data['trg']), max_seq)
    for i in range(len(data['src'])):
        for j in range(min(len(data['src'][i]), max_seq)):
            word = data['src'][i][j]
            if word in src_vocab2num:
                src_data[i][j] = src_vocab2num[word]
            else:
                src_data[i][j] = src_vocab2num['<unk>']

    for i in range(len(data['trg'])):
        for j in range(min(len(data['trg'][i]), max_seq)):
            word = data['trg'][i][j]
            if word in trg_vocab2num:
                trg_data[i][j] = trg_vocab2num[word]

            else:
                trg_data[i][j] = trg_vocab2num['<unk>']


    return src_data, trg_data


iwslt = False
batch_size = 200
print('Loading dataset')
if iwslt:
    BOS = '<s>'
    EOS = '</s>'
    PAD = "<pad>"
    SRC = Field(tokenize=tokenize_en, pad_token=PAD)
    TGT = Field(tokenize=tokenize_de, init_token=BOS, eos_token = EOS, pad_token=PAD)
    max_seq = 30
    train_data, dev_data, test_data = IWSLT.splits(exts=('.en', '.de'), fields=(SRC, TGT), filter_pred=lambda x: len(vars(x)['src']) <= max_seq and len(vars(x)['trg']) <= max_seq)

    min_freq = 2
    batch_size = 100
    SRC.build_vocab(train_data.src, min_freq=min_freq)
    TGT.build_vocab(train_data.trg, min_freq=min_freq)

    src_vocab = SRC.vocab
    trg_vocab = TGT.vocab
    src_train_data, trg_train_data = buildTensor(train_data, max_seq, src_vocab.stoi,
                                                 trg_vocab.stoi, train=True)

    src_dev_data, trg_dev_data, raw_dev_trg = buildTensor(dev_data, max_seq, src_vocab.stoi,
                                             trg_vocab.stoi, train=False, dev=True)

    src_test_data, raw_test_trg = buildTensor(test_data, max_seq, src_vocab.stoi,
                                              trg_vocab.stoi, train=False, test=True)

    src_train_data, trg_train_data = src_train_data.type(torch.long), trg_train_data.type(torch.long)
    src_dev_data, trg_dev_data = src_dev_data.type(torch.long), trg_dev_data.type(torch.long)
    src_test_data = src_test_data.type(torch.long)

    print('Building dataset')
    train = TensorDataset(src_train_data, trg_train_data)
    dev = TensorDataset(src_dev_data, trg_dev_data)
    test = TensorDataset(src_test_data)

    print('Building dataloader')
    train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True)
    dev_loader = DataLoader(dev, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, pin_memory=True)

    print('Saving dataloader')
    torch.save(src_vocab.stoi, 'src_vocab2num.pt')
    torch.save(src_vocab.itos, 'src_num2vocab.pt')
    torch.save(trg_vocab.stoi, 'trg_vocab2num.pt')
    torch.save(trg_vocab.itos, 'trg_num2vocab.pt')
    torch.save(raw_test_trg, 'raw_test_trg.pt')
    torch.save(train_loader, 'train_loader.pt')
    torch.save(dev_loader, 'dev_loader.pt')
    torch.save(test_loader, 'test_loader.pt')

else:
    src_lang = 'en'
    trg_lang = 'fr'
    train_data, dev_data, test_data = readLangs('eng', 'fra')
    train_data, src_vocab2num, trg_vocab2num, max_seq = stats_data(train_data, src_lang, trg_lang)
    dev_data = stats_data(dev_data, src_lang, trg_lang, False)
    test_data = stats_data(test_data, src_lang, trg_lang, False)
    torch.save(dev_data['trg'], 'dev_raw_trg.pt')
    torch.save(test_data['trg'], 'test_raw_trg.pt')
    print('Building dataset')
    train_src_data, train_trg_data = build_train_dev_dataset(train_data, src_vocab2num, trg_vocab2num, max_seq)
    dev_src_data, dev_trg_data = build_train_dev_dataset(dev_data, src_vocab2num, trg_vocab2num, max_seq)
    test_src_data, test_trg_data = build_train_dev_dataset(test_data, src_vocab2num, trg_vocab2num, max_seq)
    train_data = TensorDataset(train_src_data.type(torch.long), train_trg_data.type(torch.long))
    dev_data = TensorDataset(dev_src_data.type(torch.long), dev_trg_data.type(torch.long))
    test_data = TensorDataset(test_src_data.type(torch.long), test_trg_data.type(torch.long))
    print('Building dataloader')
    train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)
    torch.save(src_vocab2num, 'src_vocab2num.pt')
    torch.save(trg_vocab2num, 'trg_vocab2num.pt')

print('Saving dataloader')

torch.save(train_loader, 'train_loader.pt')
torch.save(dev_loader, 'dev_loader.pt')
torch.save(test_loader, 'test_loader.pt')
torch.save(max_seq, 'max_seq.pt')
