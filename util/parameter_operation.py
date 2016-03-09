import cPickle
import os

def saveParamsVal(path, params):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, 'wb') as f:  # open file with write-mode
        for param in params:
#             print param.get_value()
            cPickle.dump(param.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)  # serialize and save object

def loadParamsVal(path):
    toReturn = list()
    if(not os.path.exists(path)):
        return None
    with open(path, 'rb') as f:  # open file with write-mode
        while f:
            try:
                toReturn.append(cPickle.load(f))
            except:
                break
    return toReturn