from load_data import TextCorpus
import pandas as pd
#import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

#os.chdir("D:/University/Projects/AML/talk-berty-to-me") #Change path to root directory of project


def create_seq2seq_labels(text, window_size = 8):
    tokens = tokenizer(text)
    sequence_pairs = []
    for i in range(len(tokens) - window_size):
        sequence_pairs.append((tokens[i:i+window_size], tokens[i+1:i+window_size+1]))
    return sequence_pairs


def collate_batch(batch, vocab):
    titles, genres, seq_pair = zip(*batch)
    input_seq, output_seq = zip(*seq_pair)
    title_tensor = pad_sequence([torch.tensor(vocab.lookup_indices(tokenizer(t))) for t in titles],
                            batch_first=True)
    input_tensor = pad_sequence([torch.tensor(vocab.lookup_indices(t)) for t in input_seq],
                            batch_first=True)
    output_tensor = pad_sequence([torch.tensor(vocab.lookup_indices(t)) for t in output_seq],
                            batch_first=True)
    cat_tensor = pad_sequence([torch.tensor(vocab.lookup_indices(tokenizer(t))) for t in genres],
                            batch_first=True)
    return title_tensor, input_tensor, output_tensor, cat_tensor

corpus = TextCorpus('dev', True)
tokenizer = get_tokenizer("basic_english")
train_df = corpus.data
train_df.loc[:,'seq_pairs'] = train_df['text'].apply(create_seq2seq_labels)
train_df.drop(columns=['text'], inplace=True)
train_df = train_df.explode('seq_pairs').reset_index(drop=True)
vocab = corpus.vocab