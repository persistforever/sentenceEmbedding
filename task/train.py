# coding=gbk
# -*- coding: gbk -*-

from util.parameter_operation import saveParamsVal 

def train(cr, param_path, model, batchSize=5, save_freq=10, \
          shuffle=False):
    if shuffle:
        cr.shuffle()
    train_model, n_batches, clear_func = model.getTrainingFunction(cr, batchSize=batchSize)
    valid_model = model.getValidingFunction(cr)
    test_model, origin_data, true_label = model.getTestingFunction(cr)
    print "Start to train."
    epoch = 0
    n_epochs = 1000
    ite = 0
    print param_path
    while (epoch < n_epochs):
        epoch = epoch + 1
        #######################
        for i in xrange(n_batches):
            error = train_model(i)
            ite = ite + 1
            if(ite % save_freq == 0):
                print "@iter: ", ite,
                print "\tTraining Error : " , str(error),
                print "\tValid Error: ", str(valid_model()),
                # Save train_model
                print "\tSaving parameters."
                saveParamsVal(param_path, model.getParameters())
        
        print "Now testing model."
        
        cost, pred_label = test_model()
        
        print "Test Error: ", param_path, " -> ", str(cost)
        
        for o, true_l, pred_l in zip(origin_data, true_label, pred_label):
            print "%s\t|\t%s\t|\t%s" % (str(o), str(true_l), str(pred_l))
        
        
        