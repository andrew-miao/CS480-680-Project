"""
Author: Yanting Miao
This code refer to Process.py of SamLynn Evans (https://github.com/SamLynnEvans)
"""
import re
import spacy
import os
import pandas as pd
from torchtext.data import Field, TabularDataset, Iterator
from Tokenize import Token

def read_data(src_path, tgt_path, test=False):
    """
    :param src_path: the path of source language text.
    :param tgt_path: the path of target language text.
    :param test: whether to produce dataloader for test dataset. Default = False.
    :return: raw source language text, raw target language text.
    """
    raw_src_data = open(src_path).read().strip().split('\n')
    raw_tgt_data = open(tgt_path).read().strip().split('\n')
    if test:
        separate_doc_words = ['srclang', 'origlang']
        mask_words = ['<p>', '</p>', '</doc>', '</srcset>']
        src_data, tgt_data = [], []
        for sentence in raw_src_data:
            if separate_doc_words not in sentence and sentence not in mask_words:
                src_data.append(sentence[12:-6])

        for sentence in raw_tgt_data:
            if separate_doc_words not in sentence and sentence not in mask_words:
                tgt_data.append(sentence[12:-6])

    else:
        src_data = [sentence for sentence in raw_src_data]
        tgt_data = [sentence for sentence in raw_tgt_data]

    return src_data, tgt_data


def create_fields(src_lang, tgt_lang, batch_first=True):
    """
    :param src_lang: the abbreviation of source language, for example, English = 'en'.
    :param tgt_lang: the abbreviation of target language, for example, France = 'fr'.
    :param batch_first: whether to produce tensors with the batch dimension first. Default = True.
    :return: the Fields of source language and target language.
    """
    src_tokenize = Token(src_lang)
    tgt_tokenize = Token(tgt_lang)
    src_field = Field(init_token='<bos>', eos_token='<eos>', lower=True,
                      tokenize=src_tokenize.tokenizer, batch_first=batch_first)
    tgt_field = Field(init_token='<bos>', eos_token='<eos>', lower=True,
                      tokenize=tgt_tokenize.tokenizer, batch_first=batch_first)
    return src_field, tgt_field

def create_dataloader(src_path, src_lang, tgt_path, tgt_lang, batch_first=True, train=True, test=False, batch_size=4000):
    """
    :param src_path: the path of source language text.
    :param src_lang: the abbreviation of source language, for example, English = 'en'.
    :param tgt_path: the path of target language text.
    :param tgt_lang: the abbreviation of target language, for example, France = 'fr'.
    :param batch_first: whether to produce tensors with the batch dimension first. Default = True.
    :param train: whether to produce dataloader for train dataset. Default = True.
    :param test: whether to produce dataloader for test dataset. Default = False.
    :param batch_size: the size of a single batch.
    :return: padding idx of source and target, and a dataloader.
    """
    src_data, tgt_data = read_data(src_path, tgt_path, test)
    raw_data = {'src': src_data, 'tgt': tgt_data}
    df = pd.DataFrame(raw_data, columns=["src", "tgt"])
    df.to_csv("tmp.csv", index=False)
    src_field, tgt_field = create_fields(src_lang, tgt_lang, batch_first)
    data_fields = [('src', src_field), ('tgt', tgt_field)]
    dataset = TabularDataset('./tmp.csv', format='CSV', fields=data_fields)
    dataloader = Iterator(dataset, batch_size=batch_size, train=train, shuffle=True)
    os.remove('tmp.csv')
    src_pad = src_field.vocab.stoi['<pad>']
    tgt_pad = tgt_field.vocab.stoi['<pad>']
    return src_pad, tgt_pad, dataloader