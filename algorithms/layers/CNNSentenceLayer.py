from theano import tensor as T, printing
import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import numpy
import config

from layer import layer

class CNNSentenceLayer(layer):
    def __init__(self, word_embedding_dim, \
                                   size, \
                                   tparams, \
                                   prefix="cnn", \
                                   activation_function=T.nnet.sigmoid, \
                                   mode="max"):
        self.word_embedding_dim = word_embedding_dim
        self.__WBound = 0.2
        self.__MAXDIM = 10000
        self.__datatype = config.globalFloatType()
        self.sentenceW = None
        self.sentenceB = None
        self.mode = mode
        self.activation_function = activation_function
        rng = numpy.random.RandomState(23455)
        
        self.sentenceLayerNodesNum = size[0]
        self.sentenceLayerNodesSize = [size[1], size[2]]
        
        # Get sentence layer W
        self.sentenceW = theano.shared(
            numpy.asarray(
                rng.uniform(low=-self.__WBound, high=self.__WBound, \
                size=(self.sentenceLayerNodesNum, self.sentenceLayerNodesSize[0], self.sentenceLayerNodesSize[1])), \
                dtype=self.__datatype
            ),
            borrow=True,
            name=self._p(prefix, 'W')
        )
        
        # Get sentence layer b
        sentenceB0 = numpy.zeros((self.sentenceLayerNodesNum,), dtype=self.__datatype)
        self.sentenceB = theano.shared(value=sentenceB0, borrow=True, name=self._p(prefix, 'b'))
        
        if not tparams is None:
            tparams[self._p(prefix, 'W')] = self.sentenceW
            tparams[self._p(prefix, 'b')] = self.sentenceB
        
        self.params = [self.sentenceW, self.sentenceB]
        
        self.sentenceW = self.sentenceW.dimshuffle([0, 'x', 1, 2])
        self.sentenceB = self.sentenceB.dimshuffle(['x', 0 , 'x', 'x'])
        self.outputDimension = self.sentenceLayerNodesNum * (self.word_embedding_dim - self.sentenceLayerNodesSize[1] + 1) 
    
    def getOutput(self, emb):
        sentence_out = conv.conv2d(input=emb, filters=self.sentenceW, \
                                          filter_shape=(self.sentenceLayerNodesNum, 1, self.sentenceLayerNodesSize[0], self.sentenceLayerNodesSize[1]), \
                                          ) 
        
        sentence_pool = downsample.max_pool_2d(sentence_out, (self.__MAXDIM, 1), mode=self.mode, ignore_border=False)
        
        sentence_output = self.activation_function(sentence_pool + self.sentenceB)
        n_samples = emb.shape[0]
        sentenceResults = T.reshape(sentence_output, [n_samples, self.sentenceLayerNodesNum])
        
        print "sentenceLayerNodesNum", self.sentenceLayerNodesNum
        print "word_embedding_dim", self.word_embedding_dim
        print "sentenceLayerNodesSize[1]", self.sentenceLayerNodesSize[1]
        return sentenceResults
    
