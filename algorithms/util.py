import theano
from theano import tensor as T
def getError(m1, m2, errorType = "RMSE"):
    error = None
    if(errorType == "RMSE"):
        errorVector = m1 - m2
        error = T.sqr(errorVector)
    elif (errorType == "cos"):
        def coserror(a, b):
            l1a = T.sqrt(T.sum(T.sqr(a)))
            l1b = T.sqrt(T.sum(T.sqr(b)))
            d = T.dot(a, b)
            return d / l1a / l1b
        error, _ = theano.scan(fn=coserror, sequences=[m1, m2])
        error = -error
    return error