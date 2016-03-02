# coding=utf-8
# -*- coding: utf-8 -*-
from collections import OrderedDict
from theano import tensor as T, printing
import theano
import numpy
# from DocEmbeddingNNPadding import sentenceEmbeddingNN
from algorithms.algorithm import algorithm
from util import numpy_floatX
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from algorithms.layers.lstm_layer import lstm_layer
from algorithms.layers.dropout_layer import dropout_layer
from algorithms.util import getError


class lstm(algorithm):
    def __init__(self, n_words, ydim, input_params=None):
        self.options = options = {
           "dim_proj": 128,  # word embeding dimension and LSTM number of hidden units.
            "patience": 10,  # Number of epoch to wait before early stop if no progress
            "decay_c": 0.,  # Weight decay for the classifier applied to the U weights.
            "lrate": 0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
            "n_words":n_words,  # Vocabulary size
            "optimizer": self.adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
            "batch_size":16,  # The batch size during training.
            "ydim": ydim,  # The dimension of target embedding.    noise_std=0.,
            "use_dropout": True,
            "use_media_layer": True
        }
        
        # numpy paramters.
        params = self.init_params(options)
        
        # Theano paramters,
        tparams = self.init_tparams(params)
            
        trng = RandomStreams(123)
    
        # Used for dropout.
        self.use_noise = theano.shared(numpy_floatX(0.))
    
        self.x = tensor.matrix('x', dtype='int64')
        self.mask = tensor.matrix('mask', dtype=config.floatX)
        self.y = tensor.matrix('y', dtype=config.floatX)
    
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]
    
        emb = tparams['Wemb'][self.x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['dim_proj']])
        
        
        lstm_encoder = lstm_layer(emb, mask=self.mask, dim_proj=options['dim_proj'], params=tparams, prefix="lstm")
        
        proj = lstm_encoder.output
        # The average of outputs of cells is the final output of the lstm network.
        proj = (proj * self.mask[:, :, None]).sum(axis=0)
        proj = proj / self.mask.sum(axis=0)[:, None]
            
        
        if options['use_dropout']:
            proj = dropout_layer(proj, self.use_noise, trng)
    
        if options['use_media_layer']:
            proj = tensor.nnet.sigmoid(tensor.dot(proj, tparams['U']) + tparams['b'])
#         pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
        
        self.proj = proj
        self.cost = tensor.mean(getError(self.proj, self.y, errorType="cos"))
        self.tparams = tparams
        self._params = self.getParameters()
        self._setParameters(input_params)
    
    def init_params(self, options):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.
        """
        params = OrderedDict()
        # embedding
        randn = numpy.random.rand(options['n_words'],
                                  options['dim_proj'])
        params['Wemb'] = (0.01 * randn).astype(config.floatX)
        if options['use_media_layer']:
            params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                                options['ydim']).astype(config.floatX)
            params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)
        return params
        
    def init_tparams(self, params):
        tparams = OrderedDict()
        for kk, pp in params.iteritems():
            tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams
    
    def getParameters(self):
        return self.tparams.values()
    
    def getTrainFunction(self, cr, cr_scope, batchSize=10, errorType="RMSE"):
        optimizer = self.options["optimizer"]
        
        train_set_num, valid_set_num, test_set_num = cr.getSize()
        
        n_batches = (cr_scope[1] - cr_scope[0] - 1) / batchSize + 1
        train_set_batch_num = (train_set_num - 1) / batchSize + 1
        n_batches = min(n_batches, train_set_batch_num)

        
        validSet = cr.getValidSet([0, 100])
        testSet = cr.getTestSet([0, 1000])
        
        lr = tensor.scalar(name='lr')
        grads = tensor.grad(self.cost, wrt=self.tparams.values())
        f_grad_shared, f_update = optimizer(lr, self.tparams, grads,
                                    self.x, self.mask, self.y, self.cost)
        
        def update(index):
            if self.options["use_dropout"]:
                self.use_noise.set_value(1.)
            x, mask, y = cr.getTrainSet([index * batchSize, (index + 1) * batchSize])
            cost = f_grad_shared(x, mask, y)
            f_update(self.options["lrate"])
            return cost
        
        def clear_func():
            pass
        
        return update, n_batches, clear_func
    
    def getValidingFunction(self, cr):
        x, mask, y = cr.getValidSet()
        valid_function = theano.function([],
                                                                self.cost,
                                                                givens={self.x : x,
                                                                                 self.mask : mask,
                                                                                  self.y : y},
                                                                 name='valid_function')
        return valid_function
    
    def getTestingFunction(self, cr):
        x, mask, y = cr.getTestSet()
        test_function = theano.function([],
                                                                self.cost,
                                                                givens={self.x : x,
                                                                                 self.mask : mask,
                                                                                  self.y : y},
                                                                 name='valid_function')
        return test_function
    
    def getDeployFunction(self, param):
        print "Compiling computing graph."
        deploy_model = theano.function(
             [self.x, self.mask],
             [self.proj],
             allow_input_downcast=True
         )
        print "Compiled."
        def dm(x, mask):
            self.use_noise.set_value(0.)
            return deploy_model(x, mask)
        return dm
    
    def adadelta(self, lr, tparams, grads, x, mask, y, cost):
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
                                        name='adadelta_f_grad_shared')
    
        updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
    
        f_update = theano.function([lr], [], updates=ru2up + param_up,
                                   on_unused_input='ignore',
                                   name='adadelta_f_update')
    
        return f_grad_shared, f_update
