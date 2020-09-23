# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:03:59 2020

@author: willd
"""


import torch.nn as nn 
import torch
from torch.autograd import Variable
import numpy as np



class RNN_GRU(nn.Module): 
    """
    Generative or multi-class classifier RNN using the GRU cell.
    
    Params 
    ------
    input_size (int):
        Vocabulary size for initializing the embedding layer. 
        In the case for single aminoacid prediction it is 20 (or 21).
        
    embedding_size (int):
        Dimensionality of the embedding layer. 
    
    hidden_size (int):
        Dimensionality of the hidden state of the GRU. 
        
    output_size (int): 
        Dimensionality of the output. If it's in the generative mode, 
        it is the same as the input size. 
    
    input (torch.LongTensor):
        Sequence of indices.
    
    Returns
    -------
    output (torch.tensor): 
        Sequence of predictions. In the generative case it is a probability
        distribution over tokens. 
    
    last_hidden(torch.tensor): 
        Last hidden state of the GRU. 
        
    """
    
    def __init__(self, input_size, embedding_size, hidden_size,
                 output_size, n_layers = 1):
        
        super(RNN_GRU, self).__init__()
        
        self.input_size = input_size 
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_layers = n_layers 
        self.log_softmax = nn.LogSoftmax(dim= 1)
        
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        self.gru = nn.GRU(
            input_size = embedding_size,
            hidden_size= hidden_size,
            num_layers = n_layers
        )
    
        self.decoder = nn.Linear(hidden_size, output_size)

        
    def aa_encoder(self, input): 
        "Helper function to map single aminoacids to the embedding space."
        projected = self.embedding(input)
        return projected 
    
    def forward(self, input): 
        
        embedding_tensor = self.embedding(input)
        
        # shape(seq_len = len(sequence), batch_size = 1, input_size = -1)
        embedding_tensor = embedding_tensor.view(len(input), 1, self.embedding_size)

        sequence_of_hiddens, last_hidden = self.gru(embedding_tensor)
        output_rnn = sequence_of_hiddens.view(len(input), self.hidden_size)

        output = self.decoder(output_rnn)
        # LogSoftmax the output for numerical stability
        output = self.log_softmax(output)
        return output, last_hidden


class aminoacid_vocab(): 

    def __init__(self):

        aminoacid_list = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]

        self.aa_list = aminoacid_list

        self.aa_to_ix = dict(zip(aminoacid_list, np.arange(1, 21)))
        self.ix_to_aa = dict(zip(np.arange(1, 21), aminoacid_list))


    def aminoacid_to_index(self, seq):
        "Returns a list of indices (integers) from a list of aminoacids."
        
        return [self.aa_to_ix.get(aa, 0) for aa in seq]

    def index_to_aminoacid(self, ixs): 
        "Returns a list of aminoacids, given a list of their corresponding indices."
        
        return [self.ix_to_aa.get(ix, 'X') for ix in ixs]


embedding_size= 10
hidden_size = 128
n_aminoacids = 21

rnn_model = RNN_GRU(
    input_size = n_aminoacids, 
    embedding_size = embedding_size, 
    hidden_size = hidden_size, 
    output_size = n_aminoacids
)  

rnn_model.load_state_dict(torch.load('rnn_gru.pt'))
aminoacid_vocab = aminoacid_vocab()
token_to_ix = aminoacid_vocab.aminoacid_to_index



def get_hidden_state(sequence): 
    """
    Extract the last or average hidden state of a given sequence. 
    """
    
    with torch.no_grad():
        sequence_indices = Variable(torch.LongTensor(token_to_ix(sequence)))
        all_hidden, last_hidden = rnn_model(sequence_indices)
    
      
    return last_hidden.view(hidden_size).numpy()


