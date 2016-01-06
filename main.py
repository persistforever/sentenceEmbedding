from loadDialog import CorpusReader
from measure import searchNeighbour, train, loadParamsVal
import sys
if __name__ == '__main__':
    dataset = "kefu"
    data_folder = "data/" + dataset
    text_file = data_folder + "/text"
    w2v_file = data_folder + "/w2vFlat"
    stopwords_file = "data/punct"
    
    cr = CorpusReader(2, 1, text_file, stopwords_file, w2v_file)
    cr_scope = [1, 5000]
    
    param_path = None
    model = None
    
    alg = "negativeSamplingHidden"
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
    searchNeighbour(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
    if(len(sys.argv) == 1):
        searchNeighbour(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
    else:
        mode = sys.argv[1]
        if(mode == "test"):
            searchNeighbour(cr, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
        elif(mode == "train"):
            train(cr, cr_scope, dataset, data_folder, text_file, w2v_file, stopwords_file, param_path, params, model)
    print "All finished!"