# coding=utf-8
# -*- coding: utf-8 -*-
from multiprocessing.dummy import Pool as ThreadPool

import codecs
import numpy as np
from codecs import decode
import string
from util.rubbish import isRubbishSentence
import random


class CorpusReader:
    pool = ThreadPool(12)
    labels = None
    docs = None
    stopwords = None
    w2vDict = None
    # print "(570, 301)"
    minDocSentenceNum = 2
    minSentenceWordNum = 0
    __wordDim = 200
    
    def __init__(self, minDocSentenceNum, minSentenceWordNum, dataset, stopword_file, w2v_model_file):
        self.minDocSentenceNum = minDocSentenceNum
        self.minSentenceWordNum = minSentenceWordNum
        
        # Load documents
        self.docs = loadDocuments(dataset, "utf-8")
        print "document: ", len(self.docs)
        
        # Load stop words
        self.stopwords = loadStopwords(stopword_file, "gbk")
        print "stop words: ", len(self.stopwords)
        
        # Load w2v model data from file
        self.w2vDict = loadW2vModel(w2v_model_file)
        print "w2v model contains: ", len(self.w2vDict)

    def getDocNum(self):
        return len(self.labels)
    
    def getDim(self):
        return self.__wordDim
    
    def shuffle(self):
        part0 = self.docs[0::2]
        part1 = self.docs[1::2]
        pairs = zip(part0, part1)
        random.shuffle(pairs)
        self.docs = list()
        for d in pairs:
            self.docs.extend(d)
    
    def __sentence2Matrix(self, sentence, fillzeroWord=None, onlyFront=False):
        
        wordList = sentence[0]
        sentenceType = sentence[1]
        
#         for word in wordList:
#             if not word in self.w2vDict and not word in self.stopwords:
#                 print word
        
        sentenceMatrix = map(lambda word: self.w2vDict[word] if (word in self.w2vDict and word not in self.stopwords) else None, wordList)
        sentenceMatrix = filter(lambda item: not item is None, sentenceMatrix)
        
        sentenceWordNum = len(sentenceMatrix)
        if(sentenceWordNum < self.minSentenceWordNum):
            return None

        if(fillzeroWord is not None):
            if onlyFront:
                sentenceMatrix = fillzeroWord + sentenceMatrix
            else:
                sentenceMatrix = fillzeroWord + sentenceMatrix + fillzeroWord
            
        sentenceWordNum = len(sentenceMatrix)
        
        return (sentenceMatrix, sentenceWordNum, wordList, sentenceType)
    
    def __doc2Matrix(self, sentenceList, fillzeroWord=None, onlyFront=False):
        m = map(lambda s:  self.__sentence2Matrix(s, fillzeroWord, onlyFront) , sentenceList)
        m = filter(lambda item: not item is None, m)
        if(len(m) == 0):
            return None
        dialogMatrix, sentenceWordNum, sentenceTextList, sentenceTypes = zip(*m)
        dialogMatrix = list(dialogMatrix)
        sentenceWordNum = list(sentenceWordNum)
        sentenceTextList = list(sentenceTextList)
        sentenceTypes = list(sentenceTypes)
        
        dialogSentenceNum = len(dialogMatrix)
        if(dialogSentenceNum < self.minDocSentenceNum):
            return None
        
        # Merge the sentence embedding into a holistic list.
        # dialogMatrix = reduce(add, dialogMatrix, [])
        dialogMatrix0 = list()
        for d in dialogMatrix:
            dialogMatrix0.extend(d)
        
        return (dialogMatrix0, dialogSentenceNum, sentenceWordNum, sentenceTextList, sentenceTypes)
    
    def __getDataMatrix(self, scope, fillzeroWord=None, onlyFront=False):
        scope[1] = np.min([scope[1], len(self.docs)])
        if(scope[0] < 0 or scope[0] >= scope[1]):
            return None
        batch = self.docs[scope[0]:scope[1]]
        docInfo = self.pool.map(lambda b:self.__doc2Matrix(b, fillzeroWord, onlyFront), batch)
        print "Start to reduce data."
        if(len(docInfo) == 0):
            print "Lost doc: ", self.labels.items()[scope[0]:scope[1]]
            return None
        
        docInfo = filter(None, docInfo)
        if(len(docInfo) == 0):
            return None
        print "zip docInfo"
        docMatrixes, docSentenceNums, sentenceWordNums, sentenceList, sentenceTypes = zip(*docInfo)
        
        print "Merge docInfo"
        # Merge the sentence embedding into a holistic list.
        
        docMatrixes0 = list()
        sentenceWordNums0 = list()
        sentenceList0 = list()
        sentenceTypes0 = list()
        # docMatrixes = reduce(add, docMatrixes, [])
        # sentenceWordNums = reduce(add, sentenceWordNums, [])
        # sentenceList = reduce(add, sentenceList, [])
        
        for d in docMatrixes:
            docMatrixes0.extend(d)
            
        for s in sentenceWordNums:
            sentenceWordNums0.extend(s)
            
        for s in sentenceList:
            sentenceList0.extend(s)
            
        for s in sentenceTypes:
            sentenceTypes0.extend(s)
        
        docSentenceNums = [0] + list(docSentenceNums)
        sentenceWordNums0 = [0] + sentenceWordNums0
        
        docSentenceNums = np.cumsum(docSentenceNums)
        sentenceWordNums0 = np.cumsum(sentenceWordNums0)
        
        #   print docSentenceNums
        #   print sentenceWordNums
        return (docMatrixes0, docSentenceNums, sentenceWordNums0, sentenceList0, sentenceTypes0)
    
    def __findBoarder(self, docSentenceCount, sentenceWordCount):
        maxDocSentenceNum = np.max(docSentenceCount)
        maxSentenceWordNum = np.max(np.max(sentenceWordCount))
        return maxDocSentenceNum, maxSentenceWordNum
    
    def getSentenceMatrix(self, sentenceStr, sentenceType, fillzeroWord=None, onlyFront=False):
        if(fillzeroWord is not None):
            fillzeroWord = [ [0.0] * self.getDim()] * fillzeroWord 
#             fillzeroWord = [np.zeros((self.getDim(),), dtype=theano.config.floatX)] * fillzeroWord 
        words = sentenceStr.split(" ")
        return self.__sentence2Matrix((words, sentenceType, onlyFront), fillzeroWord)
    
    # Only positive scope numbers are legal.
    def getCorpus(self, scope, fillzeroWord=None, onlyFront=False):
        if(fillzeroWord is not None):
            fillzeroWord = [ [0.0] * self.getDim()] * fillzeroWord 
        return self.__getDataMatrix(scope, fillzeroWord, onlyFront)

def loadDocuments(filename, charset="utf-8"):
    f = open(filename, "r")
    docList = list()
    sentenceList = list()
    sentenceBuffer = []
    docId = 0
    lastState = "9"
    good = False
    for line0 in f:
        try:
            line = decode(line0, charset, 'ignore')
        except:
            continue
        
        tokens = line.split(u"\t")

        state = tokens[0]
        
        if len(tokens) < 2:
            if state == "9" or state == "2":
                sentence = "split_line"
            else:
                continue
        else:
            sentence = tokens[1].strip()

        if (not sentence):
            continue;
        
        if isRubbishSentence(sentence):
            continue
        
#         if ((state != "9"  or state == "2") and sentence == ""):
#             continue;
        
        if(sentence.endswith(u".")  or  sentence.endswith(u"。")  or  sentence.endswith(u"!")  \
           or sentence.endswith(u"！")   or  sentence.endswith(u"?")  \
           or sentence.endswith(u"？") or sentence.endswith(u",")   or sentence.endswith(u"，")):
            pass
        else:
            sentence = sentence + u" 。";
        
#         if(state == "1" or state == "0"):
#             if(state == lastState or lastState == "9"  or lastState == "2"):
#                 sentenceBuffer = sentenceBuffer + sentence.split(" ")
#             else:
#                 sentenceList.append((sentenceBuffer, string.atoi(lastState)))
#                 sentenceBuffer = sentence.split(" ")
#         elif(state == "9" or state == "2"):
#             if(lastState == "9" or lastState == "2"):
#                 pass
#             else:
#                 sentenceList.append((sentenceBuffer, string.atoi(lastState)))
#                 docList.append(sentenceList)
#                 sentenceBuffer = []
#                 sentenceList = list()
#         else:
#             print "Unknown state: ", line
        
        
        if(state == "1"):
            if(lastState == "1" or lastState == "9"  or lastState == "2"):
                sentenceBuffer = sentenceBuffer + sentence.split(" ")
            else:
                sentenceList.append((sentenceBuffer, string.atoi(lastState)))
                sentenceBuffer = sentence.split(" ")
                good = True
        elif(state == "0"):
            if(lastState == "0" or lastState == "9"  or lastState == "2"):
                sentenceBuffer = sentenceBuffer + sentence.split(" ")
            else:
                if good:
                    sentenceList.append((sentenceBuffer, string.atoi(lastState)))
                    docList.append(sentenceList)
#                     for s in sentenceList:
#                         print string.join(s[0], " ")
                sentenceBuffer = sentence.split(" ")
                sentenceList = list()
                good = False
        elif(state == "9" or state == "2"):
            if(lastState == "9" or lastState == "2"):
                sentenceBuffer = []
                sentenceList = list()
            else:
                if good:
                    sentenceList.append((sentenceBuffer, string.atoi(lastState)))
                    docList.append(sentenceList)
#                     for s in sentenceList:
#                         print string.join(s[0], " ")
                    
                sentenceBuffer = []
                sentenceList = list()
                good = False
        else:
            print "Unknown state: ", line
        
        lastState = state
        
    if(lastState == "9"):
        pass
    else:
        if good:
            sentenceList.append((sentenceBuffer, state))
            docList.append(sentenceList)
#             for s in sentenceList:
#                 print string.join(s[0], " ")
        sentenceBuffer = ""
        sentenceList = list()
    f.close()
    return docList

def getWords(wordsStr):
    return map(lambda word:   word[:word.index(":")], filter(lambda word: len(word) > 1 and ":" in word, wordsStr.split(" ")))

def loadStopwords(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset, "ignore")
    d = set()
    for line in f :
        d.add(line.strip("\r\n"))
    f.close()
    return d

def loadW2vModel(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset)
    d = dict()
    for line in f :
        data = line.strip("\r\n").split(" ")
        word = data[0]
        vec = map(lambda s:string.atof(s), data[1:]);
#         d[word] = np.array(vec, dtype=theano.config.floatX)
        d[word] = vec
    f.close()
    return d

if __name__ == '__main__':
    cr = CorpusReader(1, 1, "data/dialog", "data/punct", "data/dialog_w2vFlat")
    cr.getCorpus([0, 10])
