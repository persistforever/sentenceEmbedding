# coding=utf-8
# -*- coding: utf-8 -*-
from collections import OrderedDict
from theano import tensor as T, printing
import theano
import numpy
# from DocEmbeddingNNPadding import sentenceEmbeddingNN
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from util import numpy_floatX
from algorithms.layers.lstm_layer import lstm_layer
from algorithms.layers.dropout_layer import dropout_layer
from algorithms.util import getError
from algorithms.lstm import lstm

import config

class lstm_given_embedding(lstm):
    def __init__(self, hidden_dim, ydim, embedding_matrix, \
                 input_params=None, use_dropout=True,
                 activation_function=tensor.nnet.sigmoid):
        self.__embedding_matrix = embedding_matrix
        
        self.options = options = {
           "dim_proj": hidden_dim,  # word embeding dimension and LSTM number of hidden units.
            "lrate": 0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
            "optimizer": self.adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
            "ydim": ydim,  # The dimension of target embedding.    noise_std=0.,
            "use_dropout": use_dropout,
#             "use_media_layer": use_media_layer
        }
        
        # numpy paramters.
        params = self.init_params(options)
        
        # Theano paramters,
        tparams = self.init_tparams(params)
            
        trng = RandomStreams(123)
    
        # Used for dropout.
        self.use_noise = theano.shared(numpy_floatX(0.))
    
        self.x = tensor.matrix('x', dtype='int64')
        self.mask = tensor.matrix('mask', dtype=config.globalFloatType())
        self.y = tensor.matrix('y', dtype=config.globalFloatType())
    
        self.word_embedding = tensor.matrix('word_embedding', dtype=config.globalFloatType())
    
        self.all_embedding = tensor.concatenate((tparams['Wemb_special'], self.word_embedding), axis=0)
    
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]
    
        emb = self.all_embedding[self.x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['dim_proj']])
        
        proj = self.connect_layers(emb, mask=self.mask, \
                                   dim_proj=options['dim_proj'], tparams=tparams, activation_function=tensor.nnet.sigmoid)
        
#         # The average of outputs of cells is the final output of the lstm network.
#         proj = (proj * self.mask[:, :, None]).sum(axis=0)
#         proj = proj / self.mask.sum(axis=0)[:, None]
            
        proj = self.get_lstm_output(proj, self.mask)
        
        if options['use_dropout']:
            proj = dropout_layer(proj, self.use_noise, trng)
    
        proj = tensor.dot(proj, tparams['U']) + tparams['b']
#         pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
        
        self.proj = proj
        self.cost = tensor.mean(getError(self.proj, self.y, errorType="RMSE"))
        self.tparams = tparams
        self._params = self.getParameters()
        self._setParameters(input_params)
    
    def connect_layers(self, emb, mask, dim_proj, \
                       tparams, activation_function=tensor.nnet.sigmoid):
        lstm_encoder = lstm_layer(emb, mask=mask, \
                                   dim_proj=dim_proj, \
                                   params=tparams, \
                                   prefix="lstm",
                                   activation_function=tensor.nnet.sigmoid)
        proj = lstm_encoder.output
        return proj
    
    
    def init_params(self, options):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.
        """
        params = OrderedDict()
        # embedding of 4 special types.
        randn = numpy.random.rand(4,
                                  options['dim_proj'])
        params['Wemb_special'] = (0.01 * randn).astype(config.globalFloatType())
        params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.globalFloatType())
        params['b'] = numpy.zeros((options['ydim'],)).astype(config.globalFloatType())
        return params
    
    def getTrainingFunction(self, cr, batchSize=10, errorType="RMSE", batch_repeat=5):
        optimizer = self.options["optimizer"]
        
        train_set_num, valid_set_num, test_set_num = cr.getSize()
        train_set_batch_num = (train_set_num - 1) / batchSize + 1
        n_batches = train_set_batch_num
        
        lr = tensor.scalar(name='lr')
        grads = tensor.grad(self.cost, wrt=self.tparams.values())
        f_grad_shared, f_update = optimizer(lr, self.tparams, grads,
                                    self.x, self.mask, self.y, self.cost, cr.getEmbeddingMatrix())
        
        def update_function(index):
            if self.options["use_dropout"]:
                self.use_noise.set_value(1.)
            (x, mask), y, _, _ = cr.getTrainSet([index * batchSize, (index + 1) * batchSize])
            for _ in xrange(batch_repeat):
                cost = f_grad_shared(x, mask, y)
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
                                                                                 self.mask : mask,
                                                                                  self.y : y,
                                                                                  self.word_embedding: cr.getEmbeddingMatrix()},
                                                                 name='valid_function')
        return valid_function
    
    def getTestingFunction(self, cr):
        (x, mask), y, index_x, _ = cr.getTestSet()
        test_function = theano.function([],
                                                                [self.cost, self.proj],
                                                                givens={self.x : x,
                                                                                 self.mask : mask,
                                                                                  self.y : y,
                                                                                  self.word_embedding: cr.getEmbeddingMatrix()},
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
    
    def adadelta(self, lr, tparams, grads, x, mask, y, cost, word_embedding):
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
    
        f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared', givens={self.word_embedding:word_embedding})
    
        updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
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
