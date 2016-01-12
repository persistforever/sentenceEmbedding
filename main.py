from loadDialog import CorpusReader
from measure import searchNeighbour, train, chaos, loadParamsVal

import sys
if __name__ == '__main__':
    dataset = "kefu"
    data_folder = "data/" + dataset
    text_file = data_folder + "/text"
    w2v_file = data_folder + "/w2vFlat"
#     w2v_file = "data/xianliao/w2vFlat"
    stopwords_file = "data/punct"
    
    print "dataset: ", dataset
    print "data_folder: ", data_folder
    print "text_file : ", text_file
    print "w2v_file : ", w2v_file
    print "stopwords_file: ", stopwords_file
    
    cr = CorpusReader(2, 1, text_file, stopwords_file, w2v_file)
    cr_scope = [0, 1000]
    batchSize = 5
    param_path = None
    model = None
    if len(sys.argv) < 2:
        alg = "negativeSamplingMulticonvWithHidden"
    else:
        alg = sys.argv[1]
    if(alg == "averageHidden"):
        from algorithms.dialogEmbeddingSentenceHiddenAverage import sentenceEmbeddingHiddenAverage
        param_path = data_folder + "/model/average_hidden.model"
        params = loadParamsVal(param_path)
        model = sentenceEmbeddingHiddenAverage(params)
    elif(alg == "average"):
        from algorithms.dialogEmbeddingSentenceDirectAverage import sentenceEmbeddingDirectAverage
        param_path = data_folder + "/model/average.model"
        params = loadParamsVal(param_path)
        model = sentenceEmbeddingDirectAverage(params)
    elif(alg == "direct"):
        from algorithms.dialogEmbeddingSentenceDirect import sentenceEmbeddingDirect
        param_path = data_folder + "/model/direct.model"
        params = loadParamsVal(param_path)
        model = sentenceEmbeddingDirect(params)
    elif(alg == "negativeSampling"):
        from algorithms.dialogEmbeddingSentenceDirectNegativeSampling import sentenceEmbeddingDirectNegativeSampling
        param_path = data_folder + "/model/direct_negative.model"
        params = loadParamsVal(param_path)
        model = sentenceEmbeddingDirectNegativeSampling(params)
    elif(alg == "negativeSamplingHidden"):
        from algorithms.dialogEmbeddingSentenceHiddenNegativeSampling import sentenceEmbeddingHiddenNegativeSampling
        param_path = data_folder + "/model/hidden_negative.model"
        params = loadParamsVal(param_path)
        model = sentenceEmbeddingHiddenNegativeSampling(params)
    elif(alg == "negativeSamplingMulticonvWithHidden"):
        from algorithms.dialogEmbeddingSentenceMulticonvHiddenNegativeSampling import sentenceEmbeddingMulticonvHiddenNegativeSampling
        param_path = data_folder + "/model/hidden_negative_multiconv.model"
        params = loadParamsVal(param_path)
        model = sentenceEmbeddingMulticonvHiddenNegativeSampling(params)
        batchSize = 100
    
    print "param_path: ", param_path
    
    if(len(sys.argv) < 3):
        train(cr, cr_scope, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model, batchSize=batchSize)
    else:
        mode = sys.argv[2]
        print "mode: ", mode
        if(mode == "test"):
            chaos(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
        elif(mode == "searchNeighbour"):
            searchNeighbour(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
        elif(mode == "train"):
            train(cr, cr_scope, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model, batchSize=batchSize)
    print "All finished!"
