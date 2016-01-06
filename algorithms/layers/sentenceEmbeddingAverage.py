import theano
import theano.tensor.signal.downsample as downsample

class sentenceEmbeddingAverage:
    
    def __init__(self,
                          corpus,
                          docSentenceCount,
                          sentenceWordCount,
                          rng,
                          wordEmbeddingDim,
                          datatype=theano.config.floatX):
        self.__wordEmbeddingDim = wordEmbeddingDim
        self.__MAXDIM = 10000
        self.__datatype = datatype
        
        
#         t = T.and_((self.shareRandge < docSentenceCount[-1] + 1),  (self.shareRandge >= docSentenceCount[0])).nonzero()
        oneDocSentenceWordCount = sentenceWordCount[docSentenceCount[0]:docSentenceCount[-1] + 1]
        
        sentenceResults, _ = theano.scan(fn=self.__dealWithSentence,
                    non_sequences=[corpus],
                     sequences=[dict(input=oneDocSentenceWordCount, taps=[-1, -0])],
                     strict=True)
        
        self.output = sentenceResults
        self.params = []
        self.outputDimension = wordEmbeddingDim
    
    def __dealWithSentence(self, sentenceWordCount0, sentenceWordCount1, docs):
#         t = T.and_((shareRandge < sentenceWordCount1),  (shareRandge >= sentenceWordCount0)).nonzero()
        sentence = docs[sentenceWordCount0:sentenceWordCount1]
        sentence_pool = downsample.max_pool_2d(sentence, (self.__MAXDIM, 1), mode="average_exc_pad", ignore_border=False)
        sentence_embedding = sentence_pool.flatten(1)
        return sentence_embedding
