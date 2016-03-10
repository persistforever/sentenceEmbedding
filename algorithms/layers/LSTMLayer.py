from  algorithms.util import ortho_weight
import numpy
import theano
import theano.tensor as tensor
from algorithms.util import numpy_floatX
from layer import layer

import config

class LSTMLayer(layer):
    def __init__(self, dim_proj, params=None, prefix='lstm', \
                 activation_function=tensor.nnet.sigmoid):
        """
        Init the LSTM parameter:
    
        :see: init_params
        """
        self.dim_proj = dim_proj
        self.activation_function = activation_function
        self.params = params
        self.prefix = prefix
        W = numpy.concatenate([ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj)], axis=1)
        U = numpy.concatenate([ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj)], axis=1)
        b = numpy.zeros((4 * dim_proj,))
        
        # bigger forget
        b[dim_proj + 1:2 * dim_proj] = 2
        if not params is None: 
            params[self._p(prefix, 'W')] = theano.shared(W, name=self._p(prefix, 'W'))
            params[self._p(prefix, 'U')] = theano.shared(U, name=self._p(prefix, 'U'))
            params[self._p(prefix, 'b')] = theano.shared(b.astype(config.globalFloatType()), self._p(prefix, 'b'))
        self.outputDimension = dim_proj
    
    def getOutput(self, state_below, mask):
        assert mask is not None
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
    
    
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]
    
        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, self.params[self._p(self.prefix, 'U')])
            preact += x_
    
            i = self.activation_function (_slice(preact, 0, dim_proj))
            f = self.activation_function (_slice(preact, 1, dim_proj))
            o = self.activation_function (_slice(preact, 2, dim_proj))
            c = self.activation_function (_slice(preact, 3, dim_proj))
            
#             i = tensor.nnet.sigmoid(_slice(preact, 0, dim_proj))
#             f = tensor.nnet.sigmoid(_slice(preact, 1, dim_proj))
#             o = tensor.nnet.sigmoid(_slice(preact, 2, dim_proj))
#             c = tensor.tanh(_slice(preact, 3, dim_proj))
    
            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_
    
            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
    
            return h, c
    
        state_below = (tensor.dot(state_below, self.params[self._p(self.prefix, 'W')]) + 
                       self.params[self._p(self.prefix, 'b')])
    
        dim_proj = self.dim_proj
        rval, _ = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                               n_samples,
                                                               dim_proj),
                                                  tensor.alloc(numpy_floatX(0.),
                                                               n_samples,
                                                               dim_proj)],
                                    name=self._p(self.prefix, '_layers'),
                                    n_steps=nsteps)
        
        return  rval[0]

    
