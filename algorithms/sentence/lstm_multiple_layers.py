# coding=utf-8
# -*- coding: utf-8 -*-
# from DocEmbeddingNNPadding import sentenceEmbeddingNN
from lstm import lstm
from algorithms.layers.lstm_layer import lstm_layer

import theano.tensor as tensor

class lstm_multiple_layers(lstm):
    def __init__(self, n_words, hidden_dim, ydim, input_params, layers_num=4, \
                 activation_function=tensor.nnet.sigmoid):
        self.layers_num = layers_num
        lstm.__init__(self, n_words, hidden_dim, ydim, input_params, \
                      activation_function=tensor.nnet.sigmoid)
    
    def connect_layers(self, emb, mask, dim_proj, tparams, \
                        activation_function=tensor.nnet.sigmoid):
        stack_below = emb
        
        for i in xrange(self.layers_num):
            lstm_encoder = lstm_layer(stack_below, mask=self.mask, \
                                       dim_proj=dim_proj, \
                                       params=tparams, \
                                       prefix="lstm" + str(i), \
                                       activation_function=tensor.nnet.sigmoid)
            stack_below = lstm_encoder.output
            
        return stack_below
