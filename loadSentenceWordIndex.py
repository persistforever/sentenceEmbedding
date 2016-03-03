# coding=utf-8
# -*- coding: utf-8 -*-
from multiprocessing.dummy import Pool as ThreadPool

import codecs
import numpy as np
from codecs import decode
import string
from util.rubbish import isRubbishSentence
import random
import numpy
import theano
import math

class CorpusReader:
    pool = ThreadPool(12)
    docs = None
    stopwords = None
    dictionary = None
    # print "(570, 301)"
    
    sentenceStartFlagIndex = -1
    sentenceEndFlagIndex = -1
    sentenceUnkFlagIndex = -1
    
    minSentenceWordNum = 0
    maxSentenceWordNum = 100000
    
    train_set = None
    valid_set = None
    test_set = None
    
    def __init__(self, maxSentenceWordNum, minSentenceWordNum, \
                 dataset_file, stopword_file, dict_file, train_valid_test_rate=[0.999, 0.0003, 0.0007]):
        self.maxSentenceWordNum = maxSentenceWordNum
        self.minSentenceWordNum = minSentenceWordNum
        
        # Load stop words
        self.stopwords = loadStopwords(stopword_file, "gbk")
        print "stop words: ", len(self.stopwords)
        
        # Load w2v model data from file
        self.dictionary, self.dictionary_reverse = loadDictionary(dict_file, "gbk")
        self.dictionary_reverse[len(self.dictionary)] = "<BEG>"
        self.dictionary["<BEG>"] = self.sentenceStartFlagIndex = len(self.dictionary)
        self.dictionary_reverse[len(self.dictionary)] = "<END>"
        self.dictionary["<END>"] = self.sentenceEndFlagIndex = len(self.dictionary) 
        self.dictionary_reverse[len(self.dictionary)] = "<UNK>"
        self.dictionary["<UNK>"] = self.sentenceUnkFlagIndex = len(self.dictionary)
        print "dictionary size: ", len(self.dictionary)
        
        # Load documents
        self.train_set, \
        self.valid_set, \
        self.test_set = loadSentences(dataset_file,
                                  self.dictionary,
                                  self.stopwords ,
                                    maxSentenceWordNum=maxSentenceWordNum,
                                    minSentenceWordNum=minSentenceWordNum,
                                    charset="gbk",
                                    train_valid_test_rate=train_valid_test_rate)
        
        print "train_set size: ", len(self.train_set[0])
        print "valid_set size: ", len(self.valid_set[0])
        print "test_set size: ", len(self.test_set[0])

    def shuffle(self):
        pass
    
    def getDictionary(self):
        return self.dictionary, self.dictionary_reverse
    
    def getYDimension(self):
        return len(self.train_set[1][0])
    
    def getSize(self):
        """
        :return len(self.train_set), len(self.valid_set), len(self.test_set)
        """
        return len(self.train_set[0]), len(self.valid_set[0]), len(self.test_set[0])
    
    def __getSet(self, scope, x, y):
        if scope:
            scope = list(scope)
            scope[1] = np.min([scope[1], len(x)])
            if(scope[0] < 0 or scope[0] >= scope[1]):
                return None
        else:
            scope = [0, len(x)]
        
        batch_x = x[scope[0]:scope[1]]
        batch_y = numpy.matrix(y[scope[0]:scope[1]]).astype(theano.config.floatX)
        
        n_samples = len(batch_x)
        lengths = [len(s) for s in batch_x]
        
        maxlen = numpy.max(lengths)
        x_data = numpy.zeros((maxlen, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
        for idx, s in enumerate(batch_x):
            x_data[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.
            
        return x_data, x_mask, batch_y, batch_x
        
    def getTrainSet(self, scope=None):
        return self.__getSet(scope, self.train_set[0], self.train_set[1])
        
    def getValidSet(self, scope=None):
        return self.__getSet(scope, self.valid_set[0], self.valid_set[1])
        
    def getTestSet(self, scope=None):
        return self.__getSet(scope, self.test_set[0], self.test_set[1])

    def getSentenceMatrix(self, sentence):
        dictionary = self.dictionary
        
        sentence = sentence.split(" ")
        sentence = map(lambda word: dictionary[word] \
                       if (word in dictionary and word not in self.stopwords) else dictionary["<UNK>"], sentence)
        sentence = [dictionary["<BEG>"]] + sentence + [dictionary["<END>"]]
        
        x_data = numpy.transpose(numpy.asmatrix(sentence, 'int64'))
        x_mask = numpy.ones((len(sentence), 1))
        
        return x_data, x_mask

def loadSentences(filename, dictionary, stopwords, maxSentenceWordNum=100000, \
                  minSentenceWordNum=1, charset="utf-8", train_valid_test_rate=[0.999, 0.0003, 0.0007]):
    f = open(filename, "r")
    docList = list()
    for line0 in f:
        try:
            line = decode(line0, charset)
        except:
            continue
        
        tokens = line.split(u"\t")
        sentence = tokens[0].strip()
        if (not sentence):
            continue;

        sentence = sentence.split(" ")
        tokenCount = len(sentence)
        if tokenCount < minSentenceWordNum or tokenCount > maxSentenceWordNum:
            continue
        
        sentence = map(lambda word: dictionary[word] \
                       if (word in dictionary and word not in stopwords) else dictionary["<UNK>"], sentence)
        sentence = [dictionary["<BEG>"]] + sentence + [dictionary["<END>"]]
        
        
        
        sentenceEmbedding = tokens[1].strip()
        sentenceEmbedding = map(lambda x: string.atof(x), sentenceEmbedding.split(" "))
         
        docList.append((sentence, sentenceEmbedding))
        
    f.close()
    
    train_rate = int(math.floor(train_valid_test_rate[0] * len(docList)))
    valid_rate = int(math.floor((train_valid_test_rate[0] + \
                                 train_valid_test_rate[1]) * len(docList)))
    test_rate = int(math.floor((train_valid_test_rate[0] + \
                                train_valid_test_rate[1] + train_valid_test_rate[2]) * len(docList)))
    
    train_set = docList[0:train_rate]
    valid_set = docList[train_rate + 1:valid_rate]
    test_set = docList[valid_rate + 1:]
    
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x][0]))
    
    sorted_index = len_argsort(train_set)
    train_set = [train_set[i] for i in sorted_index]
    train_set_x, train_set_y = zip(*train_set)
    
    sorted_index = len_argsort(valid_set)
    valid_set = [valid_set[i] for i in sorted_index]
    valid_set_x, valid_set_y = zip(*valid_set)
    
    sorted_index = len_argsort(test_set)
    test_set = [test_set[i] for i in sorted_index]
    test_set_x, test_set_y = zip(*test_set)
    
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)


def loadStopwords(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset)
    d = set()
    for line in f :
        d.add(line.strip("\r\n"))
    f.close()
    return d

def loadDictionary(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset)
    d = dict()
    d_reverse = dict()
    for line in f :
        data = line.strip("\r\n").split("\t")
        word = data[0].strip()
        index = string.atoi(data[1])
        if word in d.keys():
            print "Word '%s' appears more than once in dict." % word
            assert False
#         d[word] = np.array(vec, dtype=theano.config.floatX)
        d[word] = index
        d_reverse[index] = word
    f.close()
    return d, d_reverse

if __name__ == '__main__':
    cr = CorpusReader(1, 1, "data/dialog", "data/punct", "data/dialog_w2vFlat")
    cr.getCorpus([0, 10])
