import os
from typing import Sized
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Sampler, DataLoader
os.chdir("D:/University/Projects/AML/talk-berty-to-me") #Change path to root directory of project

class TextCorpus():
    __slots__ = ['path', 'vocab', 'data', 'flag']
    def __init__(self, arg, vocab_exists = False):
        if arg == 'train':
            self.flag = 'train'
            self.path = 'data/train.parquet'
        elif arg == 'test':
            self.flag = 'test'
            self.path = 'data/test.parquet'
        elif arg == 'val':
            self.flag = 'val'
            self.path = 'data/val.parquet'
        elif arg == 'dev':
            self.flag = 'dev'
            self.path = 'data/dev.parquet'
        self.data = pd.read_parquet(self.path)
        if not vocab_exists:
            self.vocab = self.build_vocab()
        else:
            # self.vocab = torch.load('vocab' + self.flag + '.pth')
            self.vocab = torch.load('data/vocab.pth')
            self.vocab.set_default_index(self.vocab["<unk>"])

    def yield_tokens(self):
        tokenizer = get_tokenizer("basic_english")
        train_iter = iter(self.data.loc[:,'text'] + self.data.loc[:,'title']
                           + self.data.loc[:,'genre_one_hot'])
        for text in train_iter:
            if not isinstance(text, str):
                continue
            yield tokenizer(text)

    def build_vocab(self):
        vocab = build_vocab_from_iterator(self.yield_tokens(),
                                           specials=["<unk>"], min_freq= 1000)
        vocab.set_default_index(vocab["<unk>"])
        torch.save(vocab, 'vocab' + self.flag + '.pth')
        return vocab


class TextSampler(Sampler):
    def __init__(self, corpus_data, batch_size):
        self.corpus = corpus_data
        self.batch_size = batch_size

    def __iter__(self):
        num_batches = len(self.corpus)//self.batch_size
        for i in range(num_batches):
            for j in range(self.batch_size):
                yield (j*num_batches + i)   #From HW4

    def __len__(self):
        return (len(self.corpus)//self.batch_size)*self.batch_size

