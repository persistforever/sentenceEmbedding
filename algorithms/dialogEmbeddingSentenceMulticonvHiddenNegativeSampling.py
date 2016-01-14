# coding=utf-8
# -*- coding: utf-8 -*-
from layers.mlp import HiddenLayer
from theano import tensor as T, printing
import theano
import numpy
from  layers.sentenceEmbeddingMultiNN import SentenceEmbeddingMultiNN
from algorithms.algorithm import algorithm
import util
import config
import string

class sentenceEmbeddingMulticonvHiddenNegativeSampling(algorithm):
    
    def __init__(self, input_params=None, sentenceLayerNodesNum=[150, 120], sentenceLayerNodesSize=[(2, 200), (3, 1)], negativeLambda=1, mode="max"):
        """
        mode is in {'max', 'average_inc_pad', 'average_exc_pad', 'sum'}
        """
        rng = numpy.random.RandomState(23455)
        self._corpusWithEmbeddings = T.matrix("wordIndeices")
        self._dialogSentenceCount = T.ivector("dialogSentenceCount")
        self._sentenceWordCount = T.ivector("sentenceWordCount")
        
        # for list-type data
        self._layer0 = layer0 = SentenceEmbeddingMultiNN(self._corpusWithEmbeddings, self._dialogSentenceCount, self._sentenceWordCount, rng, wordEmbeddingDim=200, \
                                                         sentenceLayerNodesNum=sentenceLayerNodesNum, \
                                                         sentenceLayerNodesSize=sentenceLayerNodesSize,
                                                         poolingSize=[(2, 1)],
                                                         mode=mode)
        
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
        self.negativeLambda = negativeLambda
#         for p in layer1.params:
#             print p.get_value()
    
    def getTrainFunction(self, cr, cr_scope, batchSize=10, errorType="RMSE"):
        normalizationError = 0
        for p in self._params:
            normalizationError += 0.5 / batchSize * T.sum(T.square(p))
            
        isAStartSentence = T.ivector("isAStartSentence")
        iass = 1 - isAStartSentence[(self._dialogSentenceCount[0] + 1):self._dialogSentenceCount[-1]]
        
        availableIndex = iass.nonzero()
        
        error = util.getError(self._nextSentence[:-1][availableIndex], self._layer0.output[1:][availableIndex], errorType)
        errorSum = T.sum(error)
        
        errorNegative = util.getError(self._nextSentence[:-1][availableIndex], self._layer0.output[-1:0:-1][availableIndex], errorType)
        errorSumNegative = T.sum(errorNegative)
        
        learning_rate = 0.01
        
        normalizationLambda = 0.0
        negativeLambda = self.negativeLambda
#         e = errorSum - negativeLambda * errorSumNegative + normalizationLambda * normalizationError
        e = negativeLambda * errorSum / errorSumNegative + normalizationLambda * normalizationError
        
        grads = T.grad(e, self._params)
        updates = [
#             (param_i, param_i - learning_rate * grad_i / batchSize)
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self._params, grads)
        ]
        print "Loading data."
        dialogMatrixes, docSentenceNums, sentenceWordNums, sl, _ = cr.getCorpus(cr_scope, 6, onlyFront=True)
        
#         cccount = 0
#         for s in sl:
#             if(cccount % 2 == 0):
#                 print "------------------"
#             print string.join(s, "")
#             cccount += 1
        
        for i in xrange(1, len(docSentenceNums)):
            if docSentenceNums[i] - docSentenceNums[i - 1] != 2:
                raise Exception("Must only contains couple sentences for each dialog.")
        
        dialogMatrixes = algorithm.transToTensor(dialogMatrixes, config.globalFloatType())
        docSentenceNums = algorithm.transToTensor(docSentenceNums, numpy.int32)
        sentenceWordNums = algorithm.transToTensor(sentenceWordNums, numpy.int32)
        
        n_batches = (len(docSentenceNums.get_value()) - 1) / batchSize + 1
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
#         from theano import ProfileMode
#         profmode = ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
        train_model = theano.function(
             [index],
             [e, errorSum, errorSumNegative, normalizationError],
             updates=updates,
             givens={
                            self._corpusWithEmbeddings: dialogMatrixes,
                            self._dialogSentenceCount: docSentenceNums[index * batchSize: (index + 1) * batchSize + 1],
                            self._sentenceWordCount: sentenceWordNums,
                            isAStartSentence: isAStartSentenceNums
                        },
            allow_input_downcast=True,
#             mode=profmode
         )
        print "Compiled."
        def clear_memory():
            train_model.free()
            dialogMatrixes.set_value([[]])
            docSentenceNums.set_value([])
            sentenceWordNums.set_value([])
        return train_model, n_batches, clear_memory
    
    def getTestFunction(self, param):
        print "Compiling computing graph."
        deploy_model = theano.function(
             [self._corpusWithEmbeddings, self._dialogSentenceCount, self._sentenceWordCount],
             [self._layer0.output, self._nextSentence, self._layer0.output],
             allow_input_downcast=True
         )
        print "Compiled."
        return deploy_model
    
