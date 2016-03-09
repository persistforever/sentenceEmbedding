# coding=utf-8
# -*- coding: utf-8 -*-
from multiprocessing.dummy import Pool as ThreadPool

import codecs
import numpy as np
import string
from util.rubbish import isRubbishSentence
import numpy
import math
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import config

def loadStopwordsFile(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset)
    d = set()
    for line in f :
        d.add(line.strip("\r\n"))
    f.close()
    return d

def loadDictionaryFile(filename, charset="utf-8"):
    d = OrderedDict()
    d_reverse = OrderedDict()
    
    d_reverse[len(d)] = "<EMPTY>"
    d["<EMPTY>"] = len(d)
    
    d_reverse[len(d)] = "<BEG>"
    d["<BEG>"] = len(d)
    
    if not "<END>"  in d.keys():
        d_reverse[len(d)] = "<END>"
        d["<END>"] = len(d)
        
    d_reverse[len(d)] = "<UNK>"
    d["<UNK>"] = len(d)
    
    
    with codecs.open(filename, "r", charset) as f:
        for line in f :
            data = line.strip("\r\n").split("\t")
            word = data[0].strip()
            
            if not word or len(word) == 0:
                print "Get a empty line in dict file, continue."
                continue
            
            if word == "</s>":
                print "Get </s> in dict file, continue and process later."
                continue
            
            if word in d.keys():
                print "Word '%s' appears more than once in dict." % word
                assert False
    #         d[word] = np.array(vec, dtype=theano.config.floatX)
            index = len(d)
            d[word] = index
            d_reverse[index] = word
    
    return d, d_reverse

def loadWordEmbeddingsFile(filename, charset="utf-8"):
    with codecs.open(filename, "r", charset) as f:
        d = OrderedDict()
        for line in f :
            data = line.strip("\r\n").split(" ")
            word = data[0]
            if word == "</s>":
                word = "<END>"
            vec = map(lambda s:string.atof(s), data[1:]);
            d[word] = vec
    return d
  
def loadSentence(sentenceStr, minSentenceWordNum, maxSentenceWordNum, \
                 dictionary, stopwords=None):
    sentence = sentenceStr.split(" ")
    
    if isRubbishSentence(sentence):
        return None
    
    tokenCount = len(sentence)
    if tokenCount < minSentenceWordNum or tokenCount > maxSentenceWordNum:
        return None
    
    if stopwords:
        sentence = map(lambda word: dictionary[word] \
                       if (word in dictionary and word not in stopwords) else dictionary["<UNK>"], sentence)
    else:
        sentence = map(lambda word: dictionary[word] \
                       if (word in dictionary) else dictionary["<UNK>"], sentence)
    sentence = [dictionary["<BEG>"]] + sentence + [dictionary["<END>"]] 
    return sentence

def getMaskData(batch):
    n_samples = len(batch)
    lengths = [len(s) for s in batch]
    
    maxlen = numpy.max(lengths)
    data = numpy.zeros((maxlen, n_samples)).astype('int64')
    mask = numpy.zeros((maxlen, n_samples)).astype(config.globalFloatType())
    for idx, s in enumerate(batch):
        data[:lengths[idx], idx] = s
        mask[:lengths[idx], idx] = 1.
        
    return data, mask
    
def merge_dict_and_embedding(word_index_dict, word_embedding_dict):
    inter_set = set(word_embedding_dict.keys()) & set(word_index_dict.keys())
    
    word_dict = OrderedDict()
    word_dict_reverse = OrderedDict()
    
    embedding_matrx = list()
    
    zeros = [0] * len(word_embedding_dict.values()[0])
    
    word_dict_reverse[len(word_dict)] = "<EMPTY>"
    word_dict["<EMPTY>"] = len(word_dict)
    if not "<EMPTY>" in  word_embedding_dict.keys():
        embedding_matrx.append(zeros)
    else:
        embedding_matrx.append(word_embedding_dict["<EMPTY>"])
        
    word_dict_reverse[len(word_dict)] = "<BEG>"
    word_dict["<BEG>"] = len(word_dict)
    if not "<BEG>" in  word_embedding_dict.keys():
        embedding_matrx.append(zeros)
    else:
        embedding_matrx.append(word_embedding_dict["<BEG>"])
        
    word_dict_reverse[len(word_dict)] = "<END>"
    word_dict["<END>"] = len(word_dict)
    if not "<END>" in  word_embedding_dict.keys():
        embedding_matrx.append(zeros)
    else:
        embedding_matrx.append(word_embedding_dict["<END>"])
    
    word_dict_reverse[len(word_dict)] = "<UNK>"
    word_dict["<UNK>"] = len(word_dict)
    if not "<UNK>" in  word_embedding_dict.keys():
        embedding_matrx.append(zeros)
    else:
        embedding_matrx.append(word_embedding_dict["<UNK>"])
    
    
    for word in inter_set:
        if word in word_dict.keys():
            continue
        word_dict_reverse[len(word_dict)] = word
        word_dict[word] = len(word_dict)
        embedding_matrx.append(word_embedding_dict[word])
    
    return word_dict, word_dict_reverse, embedding_matrx

class CorpusReader:
    __metaclass__ = ABCMeta
    
    train_set = None
    valid_set = None
    test_set = None
    
    docs = None
    stopwords = None
    dictionary = None
    
    def __init__(self, dataset_file, stopword_file=None, dict_file=None, word_embedding_file=None, \
                 train_valid_test_rate=[0.999, 0.0003, 0.0007], charset="utf-8"):
        
        # Load stop words
        self.stopwords = loadStopwordsFile(stopword_file, charset)
        print "stop words: ", len(self.stopwords)
        
        # Load dictionary
        if not dict_file:
            raise Exception("Dictionary should exist.")
        
        if not dict_file is None:
            self.dictionary, self.dictionary_reverse = loadDictionaryFile(dict_file, charset)
            print "dictionary size: ", len(self.dictionary)
        
        # Load word embedding.
        if not word_embedding_file is None:
            self.word_embedding_dict = loadWordEmbeddingsFile(word_embedding_file, charset)
            self.dictionary, self.dictionary_reverse, self.word_embedding_matrx = \
                merge_dict_and_embedding(self.dictionary, self.word_embedding_dict)
        else:
            self.dictionary_special, self.dictionary_reverse_special, self.word_embedding_matrx_special = (None, None, None)
            
        self.train_set, \
        self.valid_set, \
        self.test_set = self.divideDataSet(self.loadData(dataset_file, charset), train_valid_test_rate)
        
        print "train_set size: ", len(self.train_set[0])
        print "valid_set size: ", len(self.valid_set[0])
        print "test_set size: ", len(self.test_set[0])   
    
    def getDictionary(self):
        return self.dictionary, self.dictionary_reverse
    
    def getEmbeddingMatrix(self):
        return numpy.matrix(self.word_embedding_matrx, dtype=config.globalFloatType())
    
    def getEmbeddingMatrixWithoutSpecialFlag(self):
        
        s = set(["<EMPTY>", "<BEG>", "<END>", "<UNK>"])
        
        return numpy.matrix(self.word_embedding_matrx[len(s):], dtype=config.globalFloatType()), s
    
    def getTrainSet(self, scope=None):
        return self.getModelInput(scope, self.train_set[0], self.train_set[1])
        
    def getValidSet(self, scope=None):
        return self.getModelInput(scope, self.valid_set[0], self.valid_set[1])
        
    def getTestSet(self, scope=None):
        return self.getModelInput(scope, self.test_set[0], self.test_set[1])
    
    def getYDimension(self):
        return len(self.train_set[1][0])
    
    def getSize(self):
        """
        :return len(self.train_set), len(self.valid_set), len(self.test_set)
        """
        return len(self.train_set[0]), len(self.valid_set[0]), len(self.test_set[0])
    
    @abstractmethod
    def loadData(self, dataset_file, charset="utf-8"):
        pass
    
    @abstractmethod
    def getModelInput(self, scope, x, y):
        """
        :return transformed x, transformed y, original x, original y
        """
        pass
    
    @abstractmethod
    def shuffle(self):
        pass
 
    def divide_data_post_process(self, train_set, valid_set, test_set):
        """
         This method is not abstract, however, the derived classes may cover this method.
        """
        pass

    def divideDataSet(self, docList, train_valid_test_rate):
        train_rate = int(math.floor(train_valid_test_rate[0] * len(docList)))
        valid_rate = int(math.floor((train_valid_test_rate[0] + \
                                     train_valid_test_rate[1]) * len(docList)))
        test_rate = int(math.floor((train_valid_test_rate[0] + \
                                    train_valid_test_rate[1] + train_valid_test_rate[2]) * len(docList)))
        
        train_set = docList[0:train_rate]
        valid_set = docList[train_rate + 1:valid_rate]
        test_set = docList[valid_rate + 1:test_rate]
    
        train_set, valid_set, test_set = self.divide_data_post_process(train_set, valid_set, test_set)
        
        train_set_x, train_set_y = zip(*train_set)
        valid_set_x, valid_set_y = zip(*valid_set)
        test_set_x, test_set_y = zip(*test_set)
        
        return (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)
 

class CorpusReaderDialogPair(CorpusReader):
    pool = ThreadPool(12)
    __wordDim = 200
    
    def __init__(self, dataset_file, minDialogSentenceNum=1, \
                 maxSentenceWordNum=1 << 31, minSentenceWordNum=1, \
                 stopword_file=None, dict_file=None, \
                 word_embedding_file=None, \
                 train_valid_test_rate=[0.999, 0.0003, 0.0007], charset="utf-8"):
        self.minDialogSentenceNum = minDialogSentenceNum
        self.maxSentenceWordNum = maxSentenceWordNum
        self.minSentenceWordNum = minSentenceWordNum
        CorpusReader.__init__(self, dataset_file, stopword_file=stopword_file, \
                              dict_file=dict_file, word_embedding_file=word_embedding_file, charset=charset)
        
    def shuffle(self):
        pass
        
    def getModelInput(self, scope, x, y):
        if scope:
            scope = list(scope)
            scope[1] = np.min([scope[1], len(x)])
            if(scope[0] < 0 or scope[0] >= scope[1]):
                return None
        else:
            scope = [0, len(x)]
        
        batch_x = x[scope[0]:scope[1]]
        batch_y = y[scope[0]:scope[1]]
        x_data, x_mask = getMaskData(batch_x)
        y_data, y_mask = getMaskData(batch_y)
        
        return (x_data, x_mask), (y_data, y_mask), batch_x, batch_y  
    

    def loadData(self, dataset_file, charset="utf-8"):
        with codecs.open(dataset_file, mode="r") as f:
            docList = list()
            for line in f:
                sentences = line.split(u"\t")
                if len(sentences) <= self.minDialogSentenceNum:
                    continue
        
                dialog = self.pool.map(lambda sentence:loadSentence(sentence, self.minSentenceWordNum, \
                                        self.maxSentenceWordNum, self.dictionary, self.stopwords), \
                                        sentences)
                
                if None in dialog: 
                    continue
                
                docList.extend(zip(dialog[1:], dialog[:-1]))
        return docList

    def divide_data_post_process(self, train_set, valid_set, test_set):
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: max(len(seq[x][0], seq[x][1])))
        
        sorted_index = len_argsort(train_set)
        train_set = [train_set[i] for i in sorted_index]
    
        sorted_index = len_argsort(valid_set)
        valid_set = [valid_set[i] for i in sorted_index]
        
        sorted_index = len_argsort(test_set)
        test_set = [test_set[i] for i in sorted_index]
        
        return train_set, valid_set, test_set

    def transformInputData(self, sentence):
        dictionary = self.dictionary
        
        sentence = sentence.split(" ")
        sentence = map(lambda word: dictionary[word] \
                       if (word in dictionary and word not in self.stopwords) else dictionary["<UNK>"], sentence)
        sentence = [dictionary["<BEG>"]] + sentence + [dictionary["<END>"]]
        
        x_data = numpy.transpose(numpy.asmatrix(sentence, 'int64'))
        x_mask = numpy.ones((len(sentence), 1))
        return (x_data, x_mask)
    
    
class  CorpusReaderSentence(CorpusReader):
    pool = ThreadPool(12)
    
    def __init__(self, dataset_file, maxSentenceWordNum=1 << 32, minSentenceWordNum=1, \
                stopword_file=None, dict_file=None, \
                 word_embedding_file=None, train_valid_test_rate=[0.999, 0.0003, 0.0007], charset="utf-8"):
        self.maxSentenceWordNum = maxSentenceWordNum
        self.minSentenceWordNum = minSentenceWordNum
        CorpusReader.__init__(self, dataset_file, stopword_file=stopword_file, \
                              dict_file=dict_file, word_embedding_file=word_embedding_file, charset=charset)
    
    def loadData(self, dataset_file, charset="utf-8"):
        with codecs.open(dataset_file, mode="r", encoding=charset) as f:
            docList = list()
            for line in f:
                tokens = line.split(u"\t")
                sentence = tokens[0].strip()
                if not sentence:
                    continue
                
                sentence = loadSentence(sentence, self.minSentenceWordNum, \
                                        self.maxSentenceWordNum, self.dictionary, self.stopwords)
                if sentence is None:
                    continue
                
                sentenceEmbedding = tokens[1].strip()
                sentenceEmbedding = map(lambda x: string.atof(x), sentenceEmbedding.split(" "))
                docList.append((sentence, sentenceEmbedding))
            return docList
    
    def shuffle(self):
        pass
    
    def getModelInput(self, scope, x, y):
        if scope:
            scope = list(scope)
            scope[1] = np.min([scope[1], len(x)])
            if(scope[0] < 0 or scope[0] >= scope[1]):
                return None
        else:
            scope = [0, len(x)]
        
        batch_y = numpy.matrix(y[scope[0]:scope[1]]).astype(config.globalFloatType())
        batch_x = x[scope[0]:scope[1]]
        
        x_data, x_mask = getMaskData(batch_x)
        return (x_data, x_mask), batch_y, batch_x, batch_y 

    def transformInputData(self, sentence):
        dictionary = self.dictionary
        
        sentence = sentence.split(" ")
        sentence = self.poolmap(lambda word: dictionary[word] \
                       if (word in dictionary and word not in self.stopwords) else dictionary["<UNK>"], sentence)
        sentence = [dictionary["<BEG>"]] + sentence + [dictionary["<END>"]]
        
        x_data = numpy.transpose(numpy.asmatrix(sentence, 'int64'))
        x_mask = numpy.ones((len(sentence), 1))
        return (x_data, x_mask)

    def divide_data_post_process(self, train_set, valid_set, test_set):
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x][0]))
        
        sorted_index = len_argsort(train_set)
        train_set = [train_set[i] for i in sorted_index]
    
        sorted_index = len_argsort(valid_set)
        valid_set = [valid_set[i] for i in sorted_index]
        
        sorted_index = len_argsort(test_set)
        test_set = [test_set[i] for i in sorted_index]
        
        return train_set, valid_set, test_set
     
if __name__ == '__main__':
    cr = CorpusReader(1, 1, "data/dialog", "data/punct", "data/dialog_w2vFlat")
    cr.getCorpus([0, 10])
