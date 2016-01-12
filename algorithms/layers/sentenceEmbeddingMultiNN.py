from theano import tensor as T, printing
import theano
import theano.tensor.signal.downsample as downsample
# import theano.tensor.signal.conv as conv
import theano.tensor.nnet.conv as conv
import numpy
import config

class SentenceEmbeddingMultiNN:
    
    def __init__(self,
                          corpus,
                          docSentenceCount,
                          sentenceWordCount,
                          rng,
                          wordEmbeddingDim,
                          sentenceLayerNodesNum=[2, 2],
                          sentenceLayerNodesSize=[(2, 2), (2, 1)],
                          poolingSize=[(2, 1)],
                          datatype=config.globalFloatType(),
                          mode="average_exc_pad"):
        self.__wordEmbeddingDim = wordEmbeddingDim
        self.__sentenceLayerNodesNum = sentenceLayerNodesNum
        self.__sentenceLayerNodesSize = sentenceLayerNodesSize
        self.__poolingSize = list(poolingSize)
        self.__WBound = 0.2
        self.__MAXDIM = 10000
        self.__datatype = datatype
        
        if len(poolingSize) != len(sentenceLayerNodesSize) - 1:
            raise Exception("Please check the size of filter list and pooling size list.") 
        
        self.sentenceW = None
        self.sentenceB = None
        self.mode = mode
        # Get sentence layer W
        self.sentenceW = list()
        
        lastNodesNum = 1
        for nodesNum, nodesSize in zip(sentenceLayerNodesNum, sentenceLayerNodesSize):
            self.sentenceW.append(theano.shared(
                numpy.asarray(
                    rng.uniform(low=-self.__WBound, high=self.__WBound, size=(nodesNum, lastNodesNum, nodesSize[0], nodesSize[1])),
                    dtype=datatype
                ),
                borrow=True
                ))
            lastNodesNum = nodesNum
        
        # Get sentence layer b
        self.sentenceB = [theano.shared(value=numpy.zeros((nodesNum,), dtype=datatype), borrow=True) \
                          for nodesNum in sentenceLayerNodesNum]
        
#         t = T.and_((self.shareRandge < docSentenceCount[-1] + 1),  (self.shareRandge >= docSentenceCount[0])).nonzero()
        oneDocSentenceWordCount = sentenceWordCount[docSentenceCount[0]:docSentenceCount[-1] + 1]
                
#         p = printing.Print("corpus")
#         corpus = p(corpus)
        self.params = self.sentenceW + self.sentenceB
        self.__poolingSize.append((self.__MAXDIM, 1))
        sentenceResults, _ = theano.scan(fn=self.__dealWithSentence,
                    non_sequences=[corpus] + self.params,
                     sequences=[dict(input=oneDocSentenceWordCount, taps=[-1, -0])],
                     strict=True)
        
        self.output = sentenceResults
        
        self.outputDimension = self.__wordEmbeddingDim
        
#         for i in xrange(len(sentenceLayerNodesSize)):
#             self.outputDimension = (self.outputDimension - sentenceLayerNodesSize[i][1] + 1) / self.__poolingSize[i][1] + 1
        
        for nodesSize, pSize in zip(sentenceLayerNodesSize, self.__poolingSize):
            self.outputDimension = (self.outputDimension - nodesSize[1] + 1 - 1) / pSize[1] + 1
            
        self.outputDimension *= sentenceLayerNodesNum[-1]
    
    def __dealWithSentence(self, sentenceWordCount0, sentenceWordCount1, docs, *karg):
        
#         p = printing.Print("sentenceWordCount0")
#         sentenceWordCount0 = p(sentenceWordCount0)  
#         p = printing.Print("sentenceWordCount1")
#         sentenceWordCount1 = p(sentenceWordCount1)  
#         t = T.and_((shareRandge < sentenceWordCount1),  (shareRandge >= sentenceWordCount0)).nonzero()
#         sentenceW = self.sentenceW
#         sentenceB = self.sentenceB
        sentenceW = karg[:len(karg) / 2]
        sentenceB = karg[len(karg) / 2:]
        poolingSize = self.__poolingSize
        sentence = docs[sentenceWordCount0:sentenceWordCount1]
        
        lastOutput = sentence.dimshuffle(['x', 'x', 0, 1])
        
#         p = printing.Print("lastOutput")
#         lastOutput = p(lastOutput)
        
        count = 0
        for w, b, poolsize in zip(sentenceW, sentenceB, poolingSize):
            sentence_out = conv.conv2d(input=lastOutput, filters=w)
            
            count += 1
#             p = printing.Print("sentence_out"+str(count))
#             sentence_out = p(sentence_out)  
            
            sentence_pool = downsample.max_pool_2d(sentence_out, poolsize, mode=self.mode, ignore_border=False)
            
#             p = printing.Print("sentence_pool"+str(count))
#             sentence_pool = p(sentence_pool)  
            
            bd = b.dimshuffle(['x', 0, 'x', 'x'])
            
#             p = printing.Print("bd"+str(count))
#             bd= p(bd)  
            
            sentence_output = T.tanh(sentence_pool + bd)
            lastOutput = sentence_output
        
        sentence_embedding = lastOutput.flatten(1)
    
#         p = printing.Print("sentence_embedding")
#         sentence_embedding = p(sentence_embedding)
        
        return sentence_embedding
