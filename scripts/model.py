from collections import Counter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torch.nn.functional as F




class Encoder_BiLSTM():
    
    '''
    Encoder class for encoding Genre and Title
    '''

    def __init__(self, vocab_size, embedding_dim,  hidden_dim, num_layers, dropout = 0.3, bidirectional = True, glove_vectors = None):
        super(Encoder_BiLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(glove_vectors, freeze= False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout = dropout, bidirectional = bidirectional)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.linear(output)
        output = self.dropout(output)
        return output, hidden




class Encoder_BiGRU():
    '''
    Encoder class for encoding context (previous sentences)
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout = 0.3, bidirectional = True, glove_vectors = None):
        super(Encoder_BiGRU, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(glove_vectors, freeze= False)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout = dropout, bidirectional = bidirectional)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        output = self.linear(output)
        output = self.dropout(output)
        return output, hidden


# class Decoder_LSTM():
#     '''
#     Decoder class that takes encoded context and auxiliary information to generate sentences
#     '''

