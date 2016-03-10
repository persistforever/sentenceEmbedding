from abc import ABCMeta, abstractmethod
class layer:
    __metaclass__ = ABCMeta
    
    def _p(self, pp, name):
        return '%s_%s' % (pp, name)
    
    @abstractmethod
    def getOutput(self, input):
        """
            :Return the output of the layer instance for a given input.
        """
