from loadDialog import CorpusReader
from measure import searchNeighbour2, train, chaos, loadParamsVal

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
    save_freq = 100
    param_path = None
    model = None
    shuffle = False
    if len(sys.argv) < 2:
        alg = "negativeSamplingShuffleMulticonvWithHiddenAverageMode"
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
        model = sentenceEmbeddingMulticonvHiddenNegativeSampling(params, sentenceLayerNodesNum=[150, 120], sentenceLayerNodesSize=[(2, 200), (3, 1)])
        batchSize = 200
        save_freq = 10
    elif(alg == "negativeSamplingShuffleMulticonvWithHidden"):
        from algorithms.dialogEmbeddingSentenceMulticonvHiddenNegativeSampling import sentenceEmbeddingMulticonvHiddenNegativeSampling
        param_path = data_folder + "/model/hidden_negative_shuffle_multiconv.model"
        params = loadParamsVal(param_path)
        model = sentenceEmbeddingMulticonvHiddenNegativeSampling(params, sentenceLayerNodesNum=[1000, 120], sentenceLayerNodesSize=[(2, 200), (3, 1)])
        batchSize = 200
        save_freq = 10
        shuffle = True
    elif(alg == "negativeSamplingShuffleMulticonvWithHiddenAverageMode"):
        from algorithms.dialogEmbeddingSentenceMulticonvHiddenNegativeSampling import sentenceEmbeddingMulticonvHiddenNegativeSampling
        param_path = data_folder + "/model/hidden_negative_shuffle_multiconv_average_mode.model"
        params = loadParamsVal(param_path)
        model = sentenceEmbeddingMulticonvHiddenNegativeSampling(params, \
                                                                 sentenceLayerNodesNum=[1000, 120], sentenceLayerNodesSize=[(2, 200), (3, 1)], mode="average_inc_pad")
        batchSize = 200
        save_freq = 10
        shuffle = True
    elif(alg == "justAverage"):
        from algorithms.dialogEmbeddingSentenceJustAverage import sentenceEmbeddingJustAverage
        param_path = data_folder + "/model/just_average.model"
        params = None
        model = sentenceEmbeddingJustAverage(params)
    
    print "param_path: ", param_path
    
    if(len(sys.argv) < 3):
#         chaos(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
        train(cr, cr_scope,param_path, model, batchSize=batchSize, save_freq=save_freq, shuffle=shuffle)
    else:
        mode = sys.argv[2]
        print "mode: ", mode
        if(mode == "test"):
            chaos(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
        elif(mode == "searchNeighbour"):
            searchNeighbour2(cr, model)
        elif(mode == "train"):
            train(cr, cr_scope, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model, batchSize=batchSize, save_freq=save_freq, shuffle=shuffle)
    print "All finished!"
