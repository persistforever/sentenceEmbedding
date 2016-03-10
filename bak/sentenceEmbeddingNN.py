from theano import tensor as T, printing
import theano
import theano.tensor.signal.downsample as downsample
import theano.tensor.signal.conv as conv
import numpy
import config

class SentenceEmbeddingNN:
    def __init__(self,
                          corpus,
                          docSentenceCount,
                          sentenceWordCount,
                          rng,
                          wordEmbeddingDim,
                          sentenceLayerNodesNum=2,
                          sentenceLayerNodesSize=(2, 2),
                          datatype=config.globalFloatType(),
                          mode="average_exc_pad"):
        self.__wordEmbeddingDim = wordEmbeddingDim
        self.__sentenceLayerNodesNum = sentenceLayerNodesNum
        self.__sentenceLayerNodesSize = sentenceLayerNodesSize
        self.__WBound = 0.2
        self.__MAXDIM = 10000
        self.__datatype = datatype
        self.sentenceW = None
        self.sentenceB = None
        self.mode = mode
        # Get sentence layer W
        self.sentenceW = theano.shared(
            numpy.asarray(
                rng.uniform(low=-self.__WBound, high=self.__WBound, size=(self.__sentenceLayerNodesNum, self.__sentenceLayerNodesSize[0], self.__sentenceLayerNodesSize[1])),
                dtype=datatype
            ),
            borrow=True
        )
        
        # Get sentence layer b
        sentenceB0 = numpy.zeros((sentenceLayerNodesNum,), dtype=datatype)
        self.sentenceB = theano.shared(value=sentenceB0, borrow=True)
        
#         t = T.and_((self.shareRandge < docSentenceCount[-1] + 1),  (self.shareRandge >= docSentenceCount[0])).nonzero()
        oneDocSentenceWordCount = sentenceWordCount[docSentenceCount[0]:docSentenceCount[-1] + 1]
        
        sentenceResults, _ = theano.scan(fn=self.__dealWithSentence,
                    non_sequences=[corpus, self.sentenceW, self.sentenceB],
                     sequences=[dict(input=oneDocSentenceWordCount, taps=[-1, -0])],
                     strict=True)
        
        self.output = sentenceResults
        self.params = [self.sentenceW, self.sentenceB]
        self.outputDimension = self.__sentenceLayerNodesNum * (self.__wordEmbeddingDim - self.__sentenceLayerNodesSize[1] + 1) 
    
    def __dealWithSentence(self, sentenceWordCount0, sentenceWordCount1, docs, sentenceW, sentenceB):
#         t = T.and_((shareRandge < sentenceWordCount1),  (shareRandge >= sentenceWordCount0)).nonzero()
        sentence = docs[sentenceWordCount0:sentenceWordCount1]
        
        sentence_out = conv.conv2d(input=sentence, filters=sentenceW)
        sentence_pool = downsample.max_pool_2d(sentence_out, (self.__MAXDIM, 1), mode=self.mode, ignore_border=False)
        
        sentence_output = T.tanh(sentence_pool + sentenceB.dimshuffle([0, 'x', 'x']))
        sentence_embedding = sentence_output.flatten(1)
        return sentence_embedding
