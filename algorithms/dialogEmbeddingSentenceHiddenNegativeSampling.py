# coding=utf-8
# -*- coding: utf-8 -*-
from layers.mlp import HiddenLayer
from theano import tensor as T, printing
import theano
import numpy
from  layers.sentenceEmbeddingNN import SentenceEmbeddingNN
from algorithms.algorithm import algorithm
import util
import config

class sentenceEmbeddingHiddenNegativeSampling(algorithm):
    
    def __init__(self, input_params=None):
        rng = numpy.random.RandomState(23455)
        self._corpusWithEmbeddings = T.matrix("wordIndeices")
        self._dialogSentenceCount = T.ivector("dialogSentenceCount")
        self._sentenceWordCount = T.ivector("sentenceWordCount")
        
        # for list-type data
        self._layer0 = layer0 = SentenceEmbeddingNN(self._corpusWithEmbeddings, self._dialogSentenceCount, self._sentenceWordCount, rng, wordEmbeddingDim=200, \
                                                         sentenceLayerNodesNum=2000, \
                                                         sentenceLayerNodesSize=[5, 200],
                                                         mode="max")
        layer1 = HiddenLayer(
            rng,
            input=layer0.output,
            n_in=layer0.outputDimension,
            n_out=layer0.outputDimension,
            activation=T.tanh
        )
        self._nextSentence = layer1.output
        self._params = layer1.params + layer0.params
        self._setParameters(input_params)
    
    def getTrainFunction(self, cr, cr_scope, batchSize=10, errorType="RMSE"):
        normalizationError = 0
        for p in self._params:
            normalizationError += 0.5 / batchSize * T.sum(T.square(p))
            
        isAStartSentence = T.ivector("isAStartSentence")
        iass = 1 - isAStartSentence[(self._dialogSentenceCount[0] + 1):self._dialogSentenceCount[-1]]
        
        availableIndex = iass.nonzero()
        
        error = util.getError(self._nextSentence[:-1][availableIndex], self._layer0.output[1:][availableIndex], errorType)
        errorNegative = util.getError(self._nextSentence[:-1][availableIndex], self._layer0.output[-1:0:-1][availableIndex], errorType)
        
        errorSum = T.sum(error)
        
        errorSumNegative = T.sum(errorNegative)
        
        learning_rate = 0.01
        
        normalizationLambda = 0.00001
        negativeLambda = 1
        e = errorSum - negativeLambda * errorSumNegative + normalizationLambda * normalizationError
        
        grads = T.grad(e, self._params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self._params, grads)
        ]
        print "Loading data."
        dialogMatrixes, docSentenceNums, sentenceWordNums, _, _ = cr.getCorpus(cr_scope, 4)
        
        for i in xrange(1, len(docSentenceNums)):
            if docSentenceNums[i] - docSentenceNums[i - 1] != 2:
                raise Exception("Must only contains couple sentences for each dialog.")
        
        dialogMatrixes = algorithm.transToTensor(dialogMatrixes, config.globalFloatType())
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
             [self._layer0.output, self._nextSentence, self._layer0.output],
             allow_input_downcast=True
         )
        print "Compiled."
        return deploy_model
    
