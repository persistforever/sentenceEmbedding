import theano
import numpy
from abc import ABCMeta, abstractmethod
class algorithm:
    __metaclass__  = ABCMeta
    def __init__(self):
        self._params = None
    
    @abstractmethod
    def getTrainFunction(self):
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
