from loadSentenceWordIndex import CorpusReader
from measure import searchNeighbour, train, chaos, loadParamsVal
from algorithms.lstm import lstm 
import sys
if __name__ == '__main__':
    dataset = "plp"
    data_folder = "data/" + dataset
    dataset_file = data_folder + "/text"
    stopwords_file = "data/punct"
    dict_file = data_folder + "/dict"
    print "data_folder: ", data_folder
    print "dataset_file: ", dataset_file
    print "stopwords_file: ", stopwords_file
    print "dict_file: ", dict_file
    
#     w2v_file = data_folder + "/w2vFlat"
#     w2v_file = "data/word2vec_flat_big"
#     w2v_file = "data/xianliao/w2vFlat"
    
#     print "text_file : ", text_file
#     print "w2v_file : ", w2v_file
    
    cr = CorpusReader(20, 1, dataset_file, stopwords_file, dict_file)
    cr_scope=[0,100000]
    batchSize = 128
    save_freq = 1
    param_path = None
    model = None
    shuffle = False
    
    if len(sys.argv) < 2:
        alg = "lstm"
    else:
        alg = sys.argv[1]
    if(alg == "lstm"):
        param_path = data_folder + "/model/lstm.model"
        params = loadParamsVal(param_path)
        model = lstm(len(cr.dictionary) + 1, cr.getYDimension(), params)
    
    print "param_path: ", param_path
    
    if(len(sys.argv) < 3):
        train(cr, cr_scope, param_path, params, model, batchSize=batchSize, save_freq=save_freq)
    else:
        mode = sys.argv[2]
        print "mode: ", mode
        if(mode == "test"):
            chaos(cr, param_path, params, model)
        elif(mode == "searchNeighbour"):
            searchNeighbour(cr, param_path, params, model)
        elif(mode == "train"):
            train(cr, param_path, params, model, batchSize=batchSize, save_freq=save_freq, shuffle=shuffle)
    print "All finished!"
