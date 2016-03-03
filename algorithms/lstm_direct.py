# coding=utf-8
# -*- coding: utf-8 -*-
from algorithms.layers.lstm_layer import lstm_layer

from lstm import lstm

class lstm_direct(lstm):    
    def get_lstm_output(self, proj, mask):
        # The last of outputs of cells is the final output of the lstm network.
        proj = proj[-1]
        return proj
    
  