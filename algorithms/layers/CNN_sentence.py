from theano import tensor as T, printing
import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import numpy
import config

def _p(pp, name):
    return '%s_%s' % (pp, name)

class CNN_sentence:
    def __init__(self, emb, \
                                   word_embedding_dim, \
                                   size, \
                                   tparams, \
                                   prefix="cnn", \
                                   activation_function=T.nnet.sigmoid, \
                                   mode="max"):
        self.__word_embedding_dim = word_embedding_dim
        self.__WBound = 0.2
        self.__MAXDIM = 10000
        self.__datatype = config.globalFloatType()
        self.sentenceW = None
        self.sentenceB = None
        self.mode = mode
        
        rng = numpy.random.RandomState(23455)
        
        sentenceLayerNodesNum = size[0]
        sentenceLayerNodesSize = [size[1], size[2]]
        
        # Get sentence layer W
        self.sentenceW = theano.shared(
            numpy.asarray(
                rng.uniform(low=-self.__WBound, high=self.__WBound, \
                size=(sentenceLayerNodesNum, sentenceLayerNodesSize[0], sentenceLayerNodesSize[1])), \
                dtype=self.__datatype
            ),
            borrow=True,
            name=_p(prefix, 'W')
        )
        tparams[_p(prefix, 'W')] = self.sentenceW
        
        # Get sentence layer b
        sentenceB0 = numpy.zeros((sentenceLayerNodesNum,), dtype=self.__datatype)
        self.sentenceB = theano.shared(value=sentenceB0, borrow=True, name=_p(prefix, 'b'))
        tparams[_p(prefix, 'b')] = self.sentenceB
        
        self.params = [self.sentenceW, self.sentenceB]
        
        self.sentenceW = self.sentenceW.dimshuffle([0, 'x', 1, 2])
        self.sentenceB = self.sentenceB.dimshuffle(['x', 0 , 'x', 'x'])
        
        sentence_out = conv.conv2d(input=emb, filters=self.sentenceW, \
                                          filter_shape=(sentenceLayerNodesNum, 1, sentenceLayerNodesSize[0], sentenceLayerNodesSize[1]), \
                                          ) 
        
        sentence_pool = downsample.max_pool_2d(sentence_out, (self.__MAXDIM, 1), mode=self.mode, ignore_border=False)
        
        sentence_output = activation_function(sentence_pool + self.sentenceB)
        n_samples = emb.shape[0]
        sentenceResults = T.reshape(sentence_output, [n_samples, sentenceLayerNodesNum])
        
        self.output = sentenceResults
        print "sentenceLayerNodesNum", sentenceLayerNodesNum
        print "word_embedding_dim", word_embedding_dim
        print "sentenceLayerNodesSize[1]", sentenceLayerNodesSize[1]
        self.outputDimension = sentenceLayerNodesNum * (word_embedding_dim - sentenceLayerNodesSize[1] + 1) 
    
