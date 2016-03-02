from  algorithms.util import ortho_weight
import numpy
import theano
from theano import config
import theano.tensor as tensor
from algorithms.util import numpy_floatX
def _p(pp, name):
    return '%s_%s' % (pp, name)

class lstm_layer:
    def __init__(self, state_below, mask,  dim_proj, params, prefix='lstm'):
        """
        Init the LSTM parameter:
    
        :see: init_params
        """
        W = numpy.concatenate([ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj),
                           ortho_weight(dim_proj)], axis=1)
        params[_p(prefix, 'W')] = theano.shared(W, name=_p(prefix, 'W'))
        U = numpy.concatenate([ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj),
                               ortho_weight(dim_proj)], axis=1)
        params[_p(prefix, 'U')] = theano.shared(U, name=_p(prefix, 'U'))
        b = numpy.zeros((4 * dim_proj,))
        params[_p(prefix, 'b')] = theano.shared(b.astype(config.floatX), _p(prefix, 'b'))
    
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
    
        assert mask is not None
    
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]
    
        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, params[_p(prefix, 'U')])
            preact += x_
    
            i = tensor.nnet.sigmoid(_slice(preact, 0, dim_proj))
            f = tensor.nnet.sigmoid(_slice(preact, 1, dim_proj))
            o = tensor.nnet.sigmoid(_slice(preact, 2, dim_proj))
            c = tensor.tanh(_slice(preact, 3, dim_proj))
    
            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_
    
            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
    
            return h, c
    
        state_below = (tensor.dot(state_below, params[_p(prefix, 'W')]) +
                       params[_p(prefix, 'b')])
    
        dim_proj = dim_proj
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                               n_samples,
                                                               dim_proj),
                                                  tensor.alloc(numpy_floatX(0.),
                                                               n_samples,
                                                               dim_proj)],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps)
        self.outputDimension = dim_proj
        self.output = rval[0]

    
