# coding=utf-8
# -*- coding: utf-8 -*-
from scipy.cluster.vq import kmeans2, whiten
import os
import string
import numpy
import codecs
import heapq
# from DocEmbeddingNNPadding import sentenceEmbeddingNN
import cPickle
from loadDialog import CorpusReader

def train(cr, cr_scope, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model):
    train_model, n_batches = model.getTrainFunction(cr, cr_scope, batchSize=5)
    
    print "Start to train."
    epoch = 0
    n_epochs = 2000
    ite = 0
    
    while (epoch < n_epochs):
        epoch = epoch + 1
        #######################
        for i in range(n_batches):
            errorNum = train_model(i)
            ite = ite + 1
            if(ite % 10 == 0):
                print
                print "@iter: ", ite
                print "Error " , param_path , ": ", errorNum
                # Save train_model
                print "Saving parameters."
                saveParamsVal(param_path, model.getParameters())
                print "Saved."
        
def searchNeighbour(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model):
    class sentenceScorePair(object):
        def __init__(self, priority, sentence):
            self.priority = priority
            self.sentence = sentence
    # [array(0.11831409498308831)]        
        def __cmp__(self, other):
            return -cmp(self.priority, other.priority)
    
    base_str = u"你 芳龄 啊"
    base_type = 0
     
    test_fun = model.getTestFunction(params)
    
    matrix, snum, _, _ = cr.getSentenceMatrix(base_str, base_type, 4)
    baseSentenceEmbedding, basePred, _ = test_fun(matrix, [0, 1], [0, snum])
    
    sQueue = []
    heapq.heapify(sQueue)
    pQueue = []
    heapq.heapify(pQueue)
    
    count = 0
    topCount = 10
    step = 1000
    while(count <= 1000000):
        corpus = cr.getCorpus([count, count + step], 4)
        if(corpus is None):
            break
        dialogMatrixes, docSentenceNums, sentenceWordNums, sentenceList, sentenceTypeList = corpus
        sentenceEmbedding, pred, average_sentence = test_fun(dialogMatrixes, docSentenceNums, sentenceWordNums)
        
        for (text, embedding, predictingEmbedding, referenceEmbedding, sentenceType) \
                    in zip(sentenceList, sentenceEmbedding, pred, average_sentence, sentenceTypeList):
            text = string.join(text, " ")
            text = codecs.encode(text, "utf-8", "ignore")
            if(sentenceType == base_type):
                sScore = numpy.sqrt(numpy.sum(numpy.square(baseSentenceEmbedding - embedding)))
                if(len(sQueue) == 0 or sScore < sQueue[0].priority):
                    flag = 1
                    for o in sQueue:
                        if(o.sentence == text):
                            flag = 0
                            break
                    if(flag == 1):
                        heapq.heappush(sQueue, sentenceScorePair(sScore, text))
                        if(len(sQueue) > topCount):
                            heapq.heappop(sQueue) 
            else:
                pScore = numpy.sqrt(numpy.sum(numpy.square(basePred - referenceEmbedding)))
                if(len(pQueue) == 0 or pScore < pQueue[0].priority):
                    flag = 1
                    for o in pQueue:
                        if(o.sentence == text):
                            flag = 0
                            break
                    if(flag == 1):
                        heapq.heappush(pQueue, sentenceScorePair(pScore, text))
                        if(len(pQueue) > topCount):
                            heapq.heappop(pQueue)         
                
            print "-----------------------------similar(RMSE)--------------------------------------------------"
            for o in sQueue:
                print o.priority, "\t", o.sentence
            print "-----------------------------prediction--------------------------------------------------"
            for o in pQueue:
                print o.priority, "\t", o.sentence
        count += step

        
def chaos(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model):
    test_fun = model.getTestFunction(params)
    
    embeddingList = list()
    sentenceList = list()
    with codecs.open("data/measure/base", "r", "utf-8", "ignore") as f:
        for line in f:
            sentence, sentence_label = line.split("\t")
            matrix, snum, _, _ = cr.getSentenceMatrix(sentence, 0, 4)
            baseSentenceEmbedding, _, _ = test_fun(matrix, [0, 1], [0, snum])
            embeddingList.append(baseSentenceEmbedding)
            sentenceList.append(sentence_label)
    
    feature_matrix = numpy.array(embeddingList)
    whitened = whiten(feature_matrix)
    cluster_num = 625
    _, cluster_labels = kmeans2(whitened, cluster_num)
    from util.entropy import relativeEntropy
    e = relativeEntropy(sentence_label, cluster_labels)
    print e
    return e
    
def saveParamsVal(path, params):
    with open(path, 'wb') as f:  # open file with write-mode
        for param in params:
#             print param.get_value()
            cPickle.dump(param.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)  # serialize and save object

def loadParamsVal(path):
    toReturn = list()
    if(not os.path.exists(path)):
        return None
    with open(path, 'rb') as f:  # open file with write-mode
        while f:
            try:
                toReturn.append(cPickle.load(f))
            except:
                break
    return toReturn

if __name__ == '__main__':
    dataset = "kefu"
    data_folder = "data/" + dataset
    text_file = data_folder + "/text"
    w2v_file = data_folder + "/w2vFlat"
    stopwords_file = "data/punct"
    
    cr = CorpusReader(2, 1, text_file, stopwords_file, w2v_file)
    cr_scope = [0, 5000]
    
    param_path = None
    model = None
    
    from algorithms.dialogEmbeddingSentenceHiddenNegativeSampling import sentenceEmbeddingHiddenNegativeSampling
    param_path = data_folder + "/model/hidden_negative.model"
    params = loadParamsVal(param_path)
    model = sentenceEmbeddingHiddenNegativeSampling(params)
    
    train(cr, cr_scope, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
