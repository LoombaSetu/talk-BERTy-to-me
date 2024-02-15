import os
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
os.chdir("D:/University/Projects/AML/talk-berty-to-me") #Change path to root directory of project

class TextCorpus():
    __slots__ = ['path', 'vocab', 'data', 'flag']
    def __init__(self, *args, vocab_exists = False):
        if 'train' in args:
            self.flag = 'train'
            self.path = 'data/train.parquet'
        elif 'test' in args:
            self.flag = 'test'
            self.path = 'data/test.parquet'
        elif 'val' in args:
            self.flag = 'val'
            self.path = 'data/val.parquet'
        elif 'dev' in args:
            self.flag = 'dev'
            self.path = 'data/dev.parquet'
        self.data = pd.read_parquet(self.path)
        if not vocab_exists:
            self.vocab = self.build_vocab()
        else:
            self.vocab = torch.load('vocab' + self.flag + '.pt')

    def yield_tokens(self):
        tokenizer = get_tokenizer("basic_english")
        train_iter = iter(self.data.loc[:,'text'] + self.data.loc[:,'title'] + self.data.loc[:,'genre_one_hot'])
        for text in train_iter:
            if type(text) is not str:
                continue
            yield tokenizer(text)

    def build_vocab(self):
        vocab = build_vocab_from_iterator(self.yield_tokens(),
                                           specials=["<unk>"], min_freq= 1000)
        vocab.set_default_index(vocab["<unk>"])
        torch.save(vocab, 'vocab' + self.flag + '.pt')
        return vocab
