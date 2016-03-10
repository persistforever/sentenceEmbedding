# coding=utf-8
# -*- coding: utf-8 -*-
from collections import OrderedDict
import theano
from theano import printing
import theano.tensor as T
from algorithms.layers.FullyConnectedLayer import FullyConnectedLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy

from algorithms.util import numpy_floatX
from algorithms.layers.CNNSentenceLayer import CNNSentenceLayer
from algorithms.layers.SoftmaxLayer import SoftmaxLayer
from algorithms.layers.LSTMLayer import LSTMLayer
import config

from algorithms.algorithm import algorithm
class lstm_single(algorithm):
    def __init__(self, hidden_dim, embedding_matrix, \
                 input_params=None, use_dropout=True,
                 activation_function=T.nnet.sigmoid):
        self.options = options = {
           "dim_proj": hidden_dim,  # word embeding dimension and LSTM number of hidden units.
            "lrate": 0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
            "optimizer": self.adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
            "use_dropout": use_dropout,
#             "use_media_layer": use_media_layer
        }
        self.__embedding_matrix = embedding_matrix
        
        # numpy paramters.
        params = self.init_params(options)
        
        # Theano paramters,
        tparams = self.init_tparams(params)
            
        trng = RandomStreams(123)
    
        # Used for dropout.
        self.use_noise = theano.shared(numpy_floatX(0.))
        self.word_embedding = T.matrix('word_embedding', dtype=config.globalFloatType())
        
        lstm_encoder = LSTMLayer(dim_proj=options["dim_proj"], \
                                   params=tparams, \
                                   prefix="lstm",
                                   activation_function=T.nnet.sigmoid)
        
        
        self.x = T.matrix('x', dtype='int64')
        self.x_mask = T.matrix('mask', dtype=config.globalFloatType())
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]
        emb_x = self.word_embedding[self.x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['word_embedding_dim']])
        self.x_sentence_embedding = lstm_encoder.getOutput(state_below=emb_x, mask=self.x_mask)[-1]

        self.y = T.matrix('y', dtype=config.globalFloatType())
        self.y_mask = T.matrix('mask', dtype=config.globalFloatType())
        n_timesteps = self.y.shape[0]
        n_samples = self.y.shape[1]
        emb_y = self.word_embedding[self.y.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['word_embedding_dim']])
        self.y_sentence_embedding = lstm_encoder.getOutput(state_below=emb_y, mask=self.y_mask)[-1]
        
        full_embedding = T.concatenate((self.x_sentence_embedding, self.y_sentence_embedding), axis=1)
        
#         p = printing.Print("full_embedding")
#         full_embedding = p(full_embedding)
        
        softmax_layer = SoftmaxLayer(n_in=options["ydim"] * 2, n_out=2, tparams=tparams, prefix="softmax")
        
        self.isPair = T.vector('isPair', dtype='int64')
        p = printing.Print("self.isPair")
        self.isPair = p(self.isPair)
        self.cost, self.y_pred = softmax_layer.negative_log_likelihood(full_embedding, self.isPair)
        
        self.tparams = tparams
        self._params = self.getParameters()
        self._setParameters(input_params)
    
    def init_params(self, options):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.
        """
        params = OrderedDict()
        return params
    
    def init_tparams(self, params):
        tparams = OrderedDict()
        for kk, pp in params.iteritems():
            tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams
    
    def getParameters(self):
        return self.tparams.values()
    
    def getTrainingFunction(self, cr, batchSize=10, errorType="RMSE", batch_repeat=5):
        optimizer = self.options["optimizer"]
        
        train_set_num, valid_set_num, test_set_num = cr.getSize()
        train_set_batch_num = (train_set_num - 1) / batchSize + 1
        n_batches = train_set_batch_num
        
        lr = T.scalar(name='lr')
        grads = T.grad(self.cost, wrt=self.tparams.values())
        f_grad_shared, f_update = optimizer(lr, self.tparams, grads, \
                                    [self.x, self.x_mask, self.y, self.y_mask, self.isPair], self.cost, \
                                    givens={self.word_embedding:self.__embedding_matrix})
        
        def update_function(index):
#             if self.options["use_dropout"]:
#                 self.use_noise.set_value(1.)
            (x0, mx0), (y0, my0), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize])
            (x1, mx1), (y1, my1), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize], shuffle=True)
            (x2, mx2), (y2, my2), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize], shuffle=True)
            (x3, mx3), (y3, my3), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize], shuffle=True)
            (x4, mx4), (y4, my4), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize], shuffle=True)
            
            x = numpy.concatenate([x0, x1, x2, x3, x4], axis=1)
            y = numpy.concatenate([y0, y1, y2, y3, y4], axis=1)
            
            mx = numpy.concatenate([mx0, mx1, mx2, mx3, mx4], axis=1)
            my = numpy.concatenate([my0, my1, my2, my3, my4], axis=1)
            
            isPair = numpy.asarray([1] * len(x0[0]) + [0] * (4 * len(x0[0])), dtype='int64')
            
            for _ in xrange(batch_repeat):
                cost = f_grad_shared(x, mx, y, my, isPair)
                f_update(self.options["lrate"])
            return cost
        
        def clear_func():
            pass
        
        return update_function, n_batches, clear_func
    
    def getValidingFunction(self, cr):
        (x, mx), (y, my), _, _ = cr.getValidSet()
        isPair = numpy.asarray([1] * len(x[0]), dtype='int64')
        valid_function = theano.function([],
                                                                self.cost,
                                                                givens={self.x : x,
                                                                                self.x_mask:mx,
                                                                                  self.y : y,
                                                                                  self.y_mask:my,
                                                                                  self.isPair:isPair,
                                                                                  self.word_embedding: self.__embedding_matrix},
                                                                 name='valid_function')
        return valid_function
    
    def getTestingFunction(self, cr):
        (x, mx), (y, my), batch_x, batch_y = cr.getTestSet()
        isPair = numpy.asarray([1] * len(x[0]), dtype='int64')
        test_function = theano.function([],
                                                                [self.cost, self.y_pred],
                                                                givens={self.x : x,
                                                                                  self.x_mask:mx,
                                                                                  self.y : y,
                                                                                  self.y_mask:my,
                                                                                  self.isPair:isPair,
                                                                                  self.word_embedding: self.__embedding_matrix},
                                                                 name='test_function')
        return test_function, zip(batch_x, batch_y), isPair
    
    # TODO!!!!! Very important
    def getDeployFunction(self, cr):
        print "Compiling computing graph."
        deploy_model = theano.function(
             [self.x, self.mask],
             [self.proj],
             givens={self.word_embedding: cr.getEmbeddingMatrix()},
             allow_input_downcast=True
         )
        print "Compiled."
        def dm(x, mask):
            self.use_noise.set_value(0.)
            return deploy_model(x, mask)
        return dm
    
