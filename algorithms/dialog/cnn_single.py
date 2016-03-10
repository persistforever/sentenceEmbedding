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

import config

from algorithms.algorithm import algorithm
from Crypto.Random.random import shuffle
from numpy import dtype
class cnn_single(algorithm):

    def __init__(self, word_embedding_dim, ydim, embedding_matrix, \
                 size, input_params=None, activation_function=T.nnet.sigmoid):
        self.__embedding_matrix = embedding_matrix
        
        self.options = options = {
           "word_embedding_dim": word_embedding_dim,  # word embeding dimension and LSTM number of hidden units.
            "lrate": 0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
            "optimizer": self.adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
            "ydim": ydim,  # The dimension of target embedding.    noise_std=0.,
            "size": size
        }
        
        # numpy paramters.
        params = self.init_params(options)
        
        # Theano paramters,
        tparams = self.init_tparams(params)
            
        trng = RandomStreams(123)
    
        # Used for dropout.
        self.use_noise = theano.shared(numpy_floatX(0.))
        self.word_embedding = T.matrix('word_embedding', dtype=config.globalFloatType())
        
        cnn_layer = CNNSentenceLayer(word_embedding_dim=options["word_embedding_dim"], \
                                   size=options["size"], \
                                   tparams=tparams, \
                                   prefix="cnn",
                                   activation_function=T.nnet.sigmoid,
                                   mode="max")
        
        hidden_layer = FullyConnectedLayer(
                rng=numpy.random.RandomState(23455),
                n_in=cnn_layer.outputDimension,
                n_out=options["ydim"],
                tparams=tparams,
                prefix="fully_conn",
                activation=T.tanh
            )
        
        self.x = T.matrix('x', dtype='int64')
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]
        emb_x = self.word_embedding[self.x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['word_embedding_dim']])
        emb_x = emb_x.dimshuffle([1, 'x', 0, 2])
        self.x_sentence_embedding = hidden_layer.getOutput(cnn_layer.getOutput(emb_x))
        
        self.y = T.matrix('y', dtype='int64')
        n_timesteps = self.y.shape[0]
        n_samples = self.y.shape[1]
        emb_y = self.word_embedding[self.y.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['word_embedding_dim']])
        emb_y = emb_y.dimshuffle([1, 'x', 0, 2])
        self.y_sentence_embedding = hidden_layer.getOutput(cnn_layer.getOutput(emb_y))
        
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
                                    [self.x, self.y, self.isPair], self.cost, \
                                    givens={self.word_embedding:self.__embedding_matrix})
        
        def update_function(index):
#             if self.options["use_dropout"]:
#                 self.use_noise.set_value(1.)
            (x0, _), (y0, _), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize])
            (x1, _), (y1, _), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize], shuffle=True)
            (x2, _), (y2, _), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize], shuffle=True)
            (x3, _), (y3, _), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize], shuffle=True)
            (x4, _), (y4, _), _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize], shuffle=True)
            
            x = numpy.concatenate([x0, x1, x2, x3, x4], axis=1)
            y = numpy.concatenate([y0, y1, y2, y3, y4], axis=1)
            
            isPair = numpy.asarray([1] * len(x0[0]) + [0] * (4 * len(x0[0])), dtype='int64')
            
            for _ in xrange(batch_repeat):
                cost = f_grad_shared(x, y, isPair)
                f_update(self.options["lrate"])
            return cost
        
        def clear_func():
            pass
        
        return update_function, n_batches, clear_func
    
    def getValidingFunction(self, cr):
        (x, _), (y, _), _, _ = cr.getValidSet()
        isPair = numpy.asarray([1] * len(x[0]), dtype='int64')
        valid_function = theano.function([],
                                                                self.cost,
                                                                givens={self.x : x,
                                                                                  self.y : y,
                                                                                  self.isPair:isPair,
                                                                                  self.word_embedding: self.__embedding_matrix},
                                                                 name='valid_function')
        return valid_function
    
    def getTestingFunction(self, cr):
        (x, _), (y, _), batch_x, batch_y = cr.getTestSet()
        isPair = numpy.asarray([1] * len(x[0]), dtype='int64')
        test_function = theano.function([],
                                                                [self.cost, self.y_pred],
                                                                givens={self.x : x,
                                                                                  self.y : y,
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
    
