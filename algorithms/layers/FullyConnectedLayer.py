"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T
from layer import layer

import config

# start-snippet-1
class FullyConnectedLayer(layer):
    def __init__(self, rng, n_in, n_out, \
                 tparams=None, prefix="fully_conn", activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        
        self.activation = activation
        
        print "%s n_in: %d" % (prefix, n_in)
        print "%s n_out: %d" % (prefix, n_out)
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=config.globalFloatType()
        )
        if activation == T.nnet.sigmoid:
            W_values *= 4

        W = theano.shared(value=W_values, name=self._p(prefix, 'W'), borrow=True)

        b_values = numpy.zeros((n_out,), dtype=config.globalFloatType())
        b = theano.shared(value=b_values, name=self._p(prefix, 'b'), borrow=True)

        self.W = W
        self.b = b
        if  not tparams is None:
            tparams[self._p(prefix, 'W')] = self.W
            tparams[self._p(prefix, 'b')] = self.b
        # parameters of the model
        self.params = [self.W, self.b]

    def getOutput(self, input_data):
        lin_output = T.dot(input_data, self.W) + self.b
        output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        return output

