# coding=utf-8
# -*- coding: utf-8 -*-
from layers.mlp import HiddenLayer
from theano import tensor as T, printing
import theano
import numpy
from  layers.sentenceEmbeddingNN import SentenceEmbeddingNN
from layers.sentenceEmbeddingAverage import sentenceEmbeddingAverage
from algorithms.algorithm import algorithm

class sentenceEmbeddingHiddenAverage(algorithm):
    
    def __init__(self, input_params=None):
        rng = numpy.random.RandomState(23455)
        self._corpusWithEmbeddings = T.matrix("wordIndeices")
        self._dialogSentenceCount = T.ivector("dialogSentenceCount")
        self._sentenceWordCount = T.ivector("sentenceWordCount")
        
        # for list-type data
        self._layer0 = layer0 = SentenceEmbeddingNN(self._corpusWithEmbeddings, self._dialogSentenceCount, self._sentenceWordCount, rng, wordEmbeddingDim=200, \
                                                         sentenceLayerNodesNum=1000, \
                                                         sentenceLayerNodesSize=[5, 200],
                                                         mode="max")
        layer1 = HiddenLayer(
            rng,
            input=layer0.output,
            n_in=layer0.outputDimension,
            n_out=200,
            activation=T.tanh
        )
        self._average_layer = sentenceEmbeddingAverage(self._corpusWithEmbeddings, self._dialogSentenceCount, self._sentenceWordCount, rng, wordEmbeddingDim=200)
        self._nextSentence = layer1.output
        self._params = layer1.params + layer0.params
        self._setParameters(input_params)
    
    def getTrainFunction(self, cr, cr_scope, batchSize=10, errorType="RMSE"):
        isAStartSentence = T.ivector("isAStartSentence")
        iass = 1 - isAStartSentence[(self._dialogSentenceCount[0] + 1):self._dialogSentenceCount[-1]]
        
        import util
        error = util.getError(self._nextSentence[:-1], self._average_layer.output[1:], errorType)
        
        error = T.dot(iass, error)
        errorSum = T.sum(error)
        
        learning_rate = 0.01
        grads = T.grad(errorSum, self._params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self._params, grads)
        ]
        print "Loading data."
        dialogMatrixes, docSentenceNums, sentenceWordNums, _, _ = cr.getCorpus(cr_scope, 4)
        dialogMatrixes = algorithm.transToTensor(dialogMatrixes, theano.config.floatX)
        docSentenceNums = algorithm.transToTensor(docSentenceNums, numpy.int32)
        sentenceWordNums = algorithm.transToTensor(sentenceWordNums, numpy.int32)
        
        n_batches = (len(docSentenceNums.get_value()) - 1) / batchSize
        isAStartSentenceNums = numpy.zeros(docSentenceNums.get_value()[-1] + 1, dtype=numpy.int32)
        for i in docSentenceNums.get_value():
            isAStartSentenceNums[i] = 1
        isAStartSentenceNums = theano.shared(
                 isAStartSentenceNums,
                borrow=True
            )
        print "Train set size is ", len(docSentenceNums.get_value()) - 1
        print "Batch size is ", batchSize
        print "Number of training batches  is ", n_batches
        print "Data loaded."
        index = T.lscalar("index")
        print "Compiling computing graph."
        train_model = theano.function(
             [index],
             [errorSum],
             updates=updates,
             givens={
                            self._corpusWithEmbeddings: dialogMatrixes,
                            self._dialogSentenceCount: docSentenceNums[index * batchSize: (index + 1) * batchSize + 1],
                            self._sentenceWordCount: sentenceWordNums,
                            isAStartSentence: isAStartSentenceNums
                        },
            allow_input_downcast=True
         )
        print "Compiled."
        return train_model, n_batches
    
    def getTestFunction(self, param):
        print "Compiling computing graph."
        deploy_model = theano.function(
             [self._corpusWithEmbeddings, self._dialogSentenceCount, self._sentenceWordCount],
             [self._layer0.output, self._nextSentence, self._average_layer.output],
             allow_input_downcast=True
         )
        print "Compiled."
        return deploy_model
    
