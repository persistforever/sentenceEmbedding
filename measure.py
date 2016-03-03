# coding=utf-8
# -*- coding: utf-8 -*-
import os
import string
import numpy
import codecs
import heapq
# from DocEmbeddingNNPadding import sentenceEmbeddingNN
import cPickle
from sklearn import metrics
import time
def train(cr, cr_scope, param_path, model, batchSize=5, save_freq=10, \
          shuffle=False):
    if shuffle:
        cr.shuffle()
    train_model, n_batches, clear_func = model.getTrainFunction(cr, cr_scope, batchSize=batchSize)
    valid_model = model.getValidingFunction(cr)
    test_x, test_y, test_model = model.getTestingFunction(cr)
    print "Start to train."
    epoch = 0
    n_epochs = 1000
    ite = 0
    
    while (epoch < n_epochs):
        epoch = epoch + 1
        #######################
        for i in range(n_batches):
            errorNum = train_model(i)
            ite = ite + 1
            if(ite % save_freq == 0):
                print
                print "@iter: ", ite
                print "Training Error : " , param_path , " -> ", str(errorNum)
                print "Valid Error: ", param_path, " -> ", str(valid_model())
                # Save train_model
                print "Saving parameters."
                saveParamsVal(param_path, model.getParameters())
                print "Saved."
        
        print "Now testing model."
        
        cost , pred_y = test_model()
        
        print "Test Error: ", param_path, " -> ", str(cost)
        
        if shuffle and epoch < n_epochs:
            cr.shuffle()
            clear_func()
            train_model, n_batches, clear_func = model.getTrainFunction(cr, cr_scope, batchSize=batchSize)
#             exit()

def vtMatch(cr, model):
    test_x, test_y, test_model = model.getTestingFunction(cr)
    _, dictionary_reverse = cr.getDictionary()
    cost , pred_y = test_model()
    print "Average error : ", cost
    print "sentence\tpredicting_embedding\ttrue_embedding"
    for t_x, t_y, p_y in zip(test_x, test_y, pred_y):
        s = string.join(map(lambda x:dictionary_reverse[x], t_x)) + "\t[ "
        s += string.join(map(lambda x:str(x), t_y.getA1()), " ") + " ]\t[ "
        s += string.join(map(lambda x:str(x), p_y), " ") + " ]"
        print s
    
def searchNeighbour(cr, model):
    class sentenceScorePair(object):
        def __init__(self, priority, sentence):
            self.priority = priority
            self.sentence = sentence
    # [array(0.11831409498308831)]        
        def __cmp__(self, other):
            return -cmp(self.priority, other.priority)
    
    base_str = u"你 芳龄 啊"
    base_type = 0
     
    test_fun = model.getTestingFunction(cr)
    
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

        
def chaos(cr, param_path, params, model, method="kmeans", output_cluser_res=True):
    test_fun = model.getTestFunction(params)
    
    embeddingList = list()
    sentence_list = list()
    
    print "Calculate embeddings."
    t0 = time.time()
    with codecs.open("data/measure/base", "r", "utf-8", "ignore") as f:
        for line in f:
            sentence, sentence_label = line.split("\t")
#             print sentence
            info = cr.getSentenceMatrix(sentence, 0, 4)
            if info is None:
                continue 
            sentence_list.append((sentence, string.atoi(sentence_label)))
            matrix, snum, _, _ = info
            baseSentenceEmbedding, _, _ = test_fun(matrix, [0, 1], [0, snum])
            embeddingList.append(baseSentenceEmbedding[0])
    
    print "Start to cluster."
    import util.cluster
    if method == "kmeans":
        cluster_labels = util.cluster.kmeans(embeddingList, 625)
    elif method == "spectral":
        cluster_labels = util.cluster.spectral(embeddingList, 625)
    t1 = time.time()
    
    if output_cluser_res:
        with codecs.open(param_path + ".measure", "w", "utf-8", "ignore") as  f:
            sentence_dict = dict()
            for sentence_and_label, cluster in zip(sentence_list, cluster_labels):
                c = sentence_dict.get(cluster)
                if c is None:
                    c = list()
                    sentence_dict[cluster] = c
                c.append(sentence_and_label)
            for c, sentence_cluster in sentence_dict.items():
                f.write("Cluster: " + str(c) + " -------------------------------------------------\n")
                for s, l in sentence_cluster:
                    f.write("%d\t%s\n" % (l, s))

    e = metrics.adjusted_mutual_info_score(zip(*sentence_list)[1], cluster_labels)  
#     e = relativeEntropy(sentence_label_list, cluster_labels) + relativeEntropy(cluster_labels, sentence_label_list)
    print "method: ", method, "\tparam path: ", param_path, "\tchaos: ", e
    print "cost time: ", t1 - t0
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
# 
# if __name__ == '__main__':
#     dataset = "kefu"
#     data_folder = "data/" + dataset
#     text_file = data_folder + "/text"
#     w2v_file = data_folder + "/w2vFlat"
#     stopwords_file = "data/punct"
#     
#     cr = CorpusReader(2, 1, text_file, stopwords_file, w2v_file)
#     cr_scope = [0, 5000]
#     
#     param_path = None
#     model = None
#     
#     from algorithms.dialogEmbeddingSentenceHiddenNegativeSampling import sentenceEmbeddingHiddenNegativeSampling
#     param_path = data_folder + "/model/hidden_negative.model"
#     params = loadParamsVal(param_path)
#     model = sentenceEmbeddingHiddenNegativeSampling(params)
# #     train(cr, cr_scope, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
#     chaos(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
