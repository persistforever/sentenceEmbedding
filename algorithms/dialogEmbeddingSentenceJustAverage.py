# coding=utf-8
# -*- coding: utf-8 -*-
from theano import tensor as T, printing
import theano
import numpy
from  layers.sentenceEmbeddingNN import SentenceEmbeddingNN
from layers.sentenceEmbeddingAverage import sentenceEmbeddingAverage
# from DocEmbeddingNNPadding import sentenceEmbeddingNN

from algorithms.algorithm import algorithm
import util
import config

class sentenceEmbeddingJustAverage(algorithm):
    def __init__(self, input_params=None):
        rng = numpy.random.RandomState(23455)
        self._corpusWithEmbeddings = T.matrix("wordIndeices")
        self._dialogSentenceCount = T.ivector("dialogSentenceCount")
        self._sentenceWordCount = T.ivector("sentenceWordCount")
        
        self._average_layer  = sentenceEmbeddingAverage(self._corpusWithEmbeddings, self._dialogSentenceCount, self._sentenceWordCount, rng, wordEmbeddingDim=200)
        
        # construct the parameter array.
        self._params = None
#         self._setParameters(input_params)
    
    def getTrainFunction(self, cr, cr_scope, batchSize=10, errorType="RMSE"):
        raise Exception("There is no train function for this class.")
    
    def getTestFunction(self, param):
        print "Compiling computing graph."
        deploy_model = theano.function(
             [self._corpusWithEmbeddings, self._dialogSentenceCount, self._sentenceWordCount],
             [self._average_layer.output, self._average_layer.output, self._average_layer.output],
             allow_input_downcast=True
         )
        print "Compiled."
        return deploy_model
