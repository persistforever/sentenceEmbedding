from task.train import train
from util.parameter_operation import loadParamsVal
from dataloader.load_data import CorpusReaderDialogPair
import sys
if __name__ == '__main__':
    dataset = "plp"
    data_folder = "data/" + dataset
    dataset_file = data_folder + "/dialog"
    stopwords_file = "data/punct"
    dict_file = data_folder + "/dict"
    charset = "gbk"
    word_embedding_file = "data/plp.word.vec.small"
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
    cr = CorpusReaderDialogPair(dataset_file, stopword_file=stopwords_file, \
                      dict_file=dict_file, word_embedding_file=word_embedding_file, \
                       train_valid_test_rate=[0.8, 0.1, 0.1], \
                       charset=charset, maxSentenceWordNum=20, minSentenceWordNum=1)
#     cr_scope = [0, 9999999999]
    batchSize = 2
    save_freq = 1
    param_path = None
    model = None
    shuffle = False
    
    if len(sys.argv) < 2:
        alg = "cnn"
    else:
        alg = sys.argv[1]
    if(alg == "cnn"):
        param_path = data_folder + "/model/dialog/cnn.model"
        params = loadParamsVal(param_path)
        from algorithms.dialog.cnn_single import cnn_single
        word_embedding_dim = 100
        model = cnn_single(word_embedding_dim=word_embedding_dim, ydim=100, \
                           embedding_matrix=cr.getEmbeddingMatrix(), \
                           size=[1024, 3, word_embedding_dim], input_params=params)
    elif(alg == "lstm"):
        param_path = data_folder + "/model/dialog/lstm.model"
        params = loadParamsVal(param_path)
        from algorithms.dialog.lstm_single import lstm_single
        word_embedding_dim = 100
        model = lstm_single(word_embedding_dim=word_embedding_dim, ydim=100, \
                           embedding_matrix=cr.getEmbeddingMatrix(), \
                           size=[1024, 3, word_embedding_dim], input_params=params)
        
        
    print "param_path: ", param_path
    if(len(sys.argv) < 3):
        train(cr, param_path, model, batchSize=batchSize, save_freq=save_freq)
    else:
        mode = sys.argv[2]
        print "mode: ", mode
        if(mode == "train"):
            train(cr, param_path, params, model, batchSize=batchSize, save_freq=save_freq, shuffle=shuffle)
    print "All finished!"
