# from measure import searchNeighbour, train, chaos, loadParamsVal, vtMatch
from task.train import train
import theano.tensor
from util.parameter_operation import loadParamsVal


import sys
if __name__ == '__main__':
    dataset = "plp"
    data_folder = "data/" + dataset
    dataset_file = data_folder + "/text"
    stopwords_file = "data/punct"
    dict_file = data_folder + "/dict"
    charset = "gbk"
    word_embedding_file = "data/plp.word.vec"
    print "data_folder: ", data_folder
    print "dataset_file: ", dataset_file
    print "stopwords_file: ", stopwords_file
    print "dict_file: ", dict_file
    print "word_embedding_file: ", word_embedding_file
    print "charset: ", charset
    
#     w2v_file = data_folder + "/w2vFlat"
#     w2v_file = "data/xianliao/w2vFlat"
    
#     print "text_file : ", text_file
#     print "w2v_file : ", w2v_file
    
    
    from dataloader.load_data import CorpusReaderSentence
    cr = CorpusReaderSentence(dataset_file, stopword_file=stopwords_file, \
                      dict_file=dict_file, word_embedding_file=word_embedding_file, \
                       train_valid_test_rate=[0.999, 0.0003, 0.0007], \
                       charset=charset, maxSentenceWordNum=20, minSentenceWordNum=1)
#     cr_scope = [0, 9999999999]
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
        param_path = data_folder + "/model/sentence/lstm.model"
        params = loadParamsVal(param_path)
        from algorithms.sentence.lstm import lstm 
        model = lstm(len(cr.dictionary) + 1, 5120, cr.getYDimension(), params)
    elif(alg == "lstm_small"):
        param_path = data_folder + "/model/sentence/lstm_small.model"
        params = loadParamsVal(param_path)
        from algorithms.sentence.lstm import lstm 
        model = lstm(len(cr.dictionary) + 1, 128, cr.getYDimension(), params, \
                      use_dropout=True, activation_function=theano.tensor.tanh)
    elif(alg == "lstm_small_given_embedding"):
        param_path = data_folder + "/model/sentence/lstm_small_given_embedding.model"
        params = loadParamsVal(param_path)
        
        embedding, specialList = cr.getEmbeddingMatrixWithoutSpecialFlag()
        if len(specialList) != 4:
            raise Exception("The number of special flags in this algorithm should just be 4.")
        
        from algorithms.sentence.lstm_given_embedding import lstm_given_embedding 
        model = lstm_given_embedding(hidden_dim=200, embedding_matrix=embedding , \
                        ydim=cr.getYDimension(), input_params=params, \
                        use_dropout=True, activation_function=theano.tensor.tanh)
    elif(alg == "lstm_direct"):
        param_path = data_folder + "/model/sentence/lstm_direct.model"
        params = loadParamsVal(param_path)
        from algorithms.sentence.lstm_direct import lstm_direct 
        model = lstm_direct(len(cr.dictionary) + 1, 128, cr.getYDimension(), params, \
                      use_dropout=True, activation_function=theano.tensor.tanh)
    elif alg == "lstm_multi":
        param_path = data_folder + "/model/sentence/lstm_multi.model"
        params = loadParamsVal(param_path)
        from algorithms.sentence.lstm_multiple_layers import lstm_multiple_layers 
        model = lstm_multiple_layers(len(cr.dictionary) + 1, 512, cr.getYDimension(), params, 4)
    elif alg == "CNN_single":
        param_path = data_folder + "/model/sentence/cnn_single.model"
        params = loadParamsVal(param_path)
        from algorithms.sentence.cnn_single import cnn_single 
        model = cnn_single(word_embedding_dim=200, ydim=100, \
                           embedding_matrix=cr.getEmbeddingMatrix(), size=[1024, 3, 200], input_params=params)
    
    print "param_path: ", param_path
    if(len(sys.argv) < 3):
        train(cr, param_path, model, batchSize=batchSize, save_freq=save_freq)
    else:
        mode = sys.argv[2]
        print "mode: ", mode
        if(mode == "train"):
            train(cr, param_path, params, model, batchSize=batchSize, save_freq=save_freq, shuffle=shuffle)
    print "All finished!"
