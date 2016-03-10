import theano
import numpy
from abc import ABCMeta, abstractmethod

from algorithms.util import numpy_floatX
import theano.tensor as T

class algorithm:
    __metaclass__  = ABCMeta
    def __init__(self):
        self._params = None
    
    @abstractmethod
    def getTrainingFunction(self):
        """
            :Return a theano function, which is a training fucntion whose
            input value is a index indicates the serial number of input mini-batch.
        """
        pass
    
    @abstractmethod
    def getValidingFunction(self):
        """
            :Return a theano function which works on the valid data. The output of this fuction is similar 
            with @getTrainFunction, but without updating operation."""
        pass
    
    @abstractmethod
    def getTestingFunction(self):
        """
            :Return a theano function which works on the test data. The output of this fuction is similar 
            with @getTrainFunction, but without updating operation."""
        pass

    @abstractmethod
    def getDeployFunction(self, param):
        """
            :Return a theano function, which is a testing function. Its 
            return value is (sentence embedding, predicting next sentence embedding, reference sentence embedding).
            In general, if the predicting next  embedding of sentence A is similar to the reference sentence 
            embedding of sentence B, we say that B is approximately next to A. """
        pass
    
    def _setParameters(self, params):
        if(params is not None):
            for para0, para in zip(self._params, params):
                para0.set_value(para, borrow=True)
            
    def getParameters(self):
        return self._params
    
    @classmethod
    def transToTensor(cls, data, t):
        return theano.shared(
            numpy.array(
                data,
                dtype=t
            ),
            borrow=True
        )

    def adadelta(self, lr, tparams, grads, model_input, cost, givens):
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
        input: Theano variable of input, list.
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
    
        f_grad_shared = theano.function(model_input, cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared', givens=givens)
    
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
                                   givens=givens)
    
        return f_grad_shared, f_update
