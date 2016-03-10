# coding=utf-8
# -*- coding: utf-8 -*-
from collections import OrderedDict
import theano
import theano.tensor as T
from algorithms.layers.FullyConnectedLayer import FullyConnectedLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy

from algorithms.util import numpy_floatX
from algorithms.layers.CNNSentenceLayer import CNNSentenceLayer
from algorithms.layers.DropoutLayer import DropoutLayer
from algorithms.util import getError

import config
from algorithms.algorithm import algorithm
class cnn_single(algorithm):

    def __init__(self, word_embedding_dim, ydim, embedding_matrix, \
                 size, input_params=None, use_dropout=True, activation_function=T.nnet.sigmoid):
        self.__embedding_matrix = embedding_matrix
        
        self.options = options = {
           "word_embedding_dim": word_embedding_dim,  # word embeding dimension and LSTM number of hidden units.
            "lrate": 0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
            "optimizer": self.adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
            "ydim": ydim,  # The dimension of target embedding.    noise_std=0.,
            "use_dropout": use_dropout,
            "size": size
        }
        
        # numpy paramters.
        params = self.init_params(options)
        
        # Theano paramters,
        tparams = self.init_tparams(params)
            
        trng = RandomStreams(123)
    
        # Used for dropout.
        self.use_noise = theano.shared(numpy_floatX(0.))
    
        self.x = T.matrix('x', dtype='int64')
#         self.mask = tensor.matrix('mask', dtype=config.globalFloatType())
        self.y = T.matrix('y', dtype=config.globalFloatType())
    
        self.word_embedding = T.matrix('word_embedding', dtype=config.globalFloatType())
    
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]
    
        emb = self.word_embedding[self.x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['word_embedding_dim']])
        
        
        proj = self.connect_layers(options=options, emb=emb, \
                                   tparams=tparams, \
                                   activation_function=activation_function)
        
        if options['use_dropout']:
            drop_layer = DropoutLayer()
            proj = drop_layer.getOutput(proj, self.use_noise, trng)
    
        self.proj = proj
        self.cost = T.mean(getError(self.proj, self.y, errorType="RMSE"))
        self.tparams = tparams
        self._params = self.getParameters()
        self._setParameters(input_params)
    
    def connect_layers(self, options, emb, \
                       tparams, activation_function=T.nnet.sigmoid,):
        emb = emb.dimshuffle([1, 'x', 0, 2])
        cnn_layer = CNNSentenceLayer(word_embedding_dim=options["word_embedding_dim"], \
                                   size=options["size"], \
                                   tparams=tparams, \
                                   prefix="cnn",
                                   activation_function=T.nnet.sigmoid,
                                   mode="max")
        cnn = cnn_layer.getOutput(emb)
        
        layer1 = FullyConnectedLayer(
                rng=numpy.random.RandomState(23455),
                n_in=cnn_layer.outputDimension,
                n_out=options["ydim"],
                tparams=tparams,
                prefix="fully_conn",
                activation=T.tanh
            )
        
        return layer1.getOutput(cnn)
    
    
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
        f_grad_shared, f_update = optimizer(lr, self.tparams, grads,
                                    self.x, self.y, self.cost, cr.getEmbeddingMatrix())
        
        def update_function(index):
            if self.options["use_dropout"]:
                self.use_noise.set_value(1.)
            (x, mask), y, _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize])
            for _ in xrange(batch_repeat):
                cost = f_grad_shared(x, y)
                f_update(self.options["lrate"])
            return cost
        
        def clear_func():
            pass
        
        return update_function, n_batches, clear_func
    
    def getValidingFunction(self, cr):
        (x, mask), y, _, _ = cr.getValidSet()
        valid_function = theano.function([],
                                                                self.cost,
                                                                givens={self.x : x,
                                                                                  self.y : y,
                                                                                  self.word_embedding: self.__embedding_matrix},
                                                                 name='valid_function')
        return valid_function
    
    def getTestingFunction(self, cr):
        (x, mask), y, index_x, _ = cr.getTestSet()
        test_function = theano.function([],
                                                                [self.cost, self.proj],
                                                                givens={self.x : x,
                                                                                  self.y : y,
                                                                                  self.word_embedding: self.__embedding_matrix},
                                                                 name='valid_function')
        return test_function, (index_x, y)
    
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
    
    def adadelta(self, lr, tparams, grads, x, y, cost, word_embedding):
        """
        An adaptive learning rate optimizer
    
        Parameters
        ----------
        lr : Theano SharedVariable
            Initial learning rate
        tpramas: Theano SharedVariable
            Model parameters
        grads: Theano variable
            Gradients of cost w.r.t to parameres
        x: Theano variable
            Model inputs
        mask: Theano variable
            Sequence mask
        y: Theano variable
            Targets
        cost: Theano variable
            Objective fucntion to minimize
    
        Notes
        -----
        For more information, see [ADADELTA]_.
    
        .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
           Rate Method*, arXiv:1212.5701.
        """
    
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                      name='%s_grad' % k)
                        for k, p in tparams.iteritems()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                     name='%s_rup2' % k)
                       for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.iteritems()]
    
        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]
    
        f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared', givens={self.word_embedding:word_embedding})
    
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
    
        f_update = theano.function([lr], [], updates=ru2up + param_up,
                                   on_unused_input='ignore',
                                   name='adadelta_f_update',
                                   givens={self.word_embedding:word_embedding})
    
        return f_grad_shared, f_update
