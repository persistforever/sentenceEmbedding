# coding=utf-8
# -*- coding: utf-8 -*-
# from DocEmbeddingNNPadding import sentenceEmbeddingNN
from algorithms.lstm import lstm
from algorithms.layers.lstm_layer import lstm_layer

class lstm_multiple_layers(lstm):
    def __init__(self, n_words, hidden_dim, ydim, input_params, layers_num=4):
        self.layers_num = layers_num
        lstm.__init__(self, n_words, hidden_dim, ydim, input_params)
    
    def connect_layers(self, emb, mask, dim_proj, tparams):
        stack_below = emb
        
        for i in xrange(self.layers_num):
            lstm_encoder = lstm_layer(stack_below, mask=self.mask, \
                                       dim_proj=dim_proj, \
                                       params=tparams, \
                                       prefix="lstm" + str(i))
            stack_below = lstm_encoder.output
            
        return stack_below
