import numpy

import theano
from theano import tensor as T, printing
from numpy import dtype
from theano.ifelse import ifelse

class RNNSentence(object):
    ''' elman neural net model '''
    def __init__(self, dataset, docSentenceCount, numberHiddenNodes, dimensionSentence,
                           isAStartSentence, datatype=theano.config.floatX):
        
        self.outputDimension = numberHiddenNodes
        
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                [dimensionSentence, numberHiddenNodes]).astype(datatype))
        
        self.wh = theano.shared(name='wx',
                        value=0.2 * numpy.random.uniform(-1.0, 1.0,
                        [numberHiddenNodes, numberHiddenNodes]).astype(datatype))

        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(numberHiddenNodes,
                                dtype=datatype))
        
        self.params = [self.wx, self.wh, self.bh]
        
        self.h0 = theano.shared(name='h0',
                        value=numpy.zeros((1, numberHiddenNodes),
                        dtype=datatype))
        
#         self.shareRandge = T.arange(maxRandge)
#         t = T.and_((self.shareRandge < docSentenceCount[-1]),  (self.shareRandge >= docSentenceCount[0])).nonzero()
        isAStartSentence = isAStartSentence[docSentenceCount[0]:docSentenceCount[-1]]
        
        h, _ = theano.scan(fn=self.__recurrence,
                non_sequences=[self.wx, self.wh, self.bh, self.h0],
                sequences=[dataset, isAStartSentence],
                outputs_info=[self.h0],
                strict=True)
        
        h = h[:, 0, :]
#         p = printing.Print('h')
#         h = p(h)
        self.output = h
        
    def __recurrence(self, x_t, isAStart, h_tm1, _wx, _wh, _bh, _h0):
        h_tm1 = ifelse(T.eq(isAStart, 1), _h0, h_tm1)
        h_t = T.nnet.sigmoid(T.dot(x_t, _wx)
                             + T.dot(h_tm1, _wh) + _bh)
        return h_t 
