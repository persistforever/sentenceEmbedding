import theano.tensor as tensor
from layer import layer

class DropoutLayer(layer):
    def __init__(self):
        pass
    
    def getOutput(self, state_before, use_noise, trng):
        proj = tensor.switch(use_noise,
                             (state_before * 
                              trng.binomial(state_before.shape,
                                            p=0.5, n=1,
                                            dtype=state_before.dtype)),
                             state_before * 0.5)
        return proj
