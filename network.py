"""Initialize the Network class"""



import layer

class Network:
    """Class used to initialize a multilayer perceptron"""
    def __init__(self, layers):
        self.layers = layers

    def append(self, N, activation):
        """Append a layer of N neurons with specified activation to the network"""
        #check that there is a layer before
        if(self.layers.shape[0] != 0):
            lay = layer.Layer(N, activation, self.layers[0].shape[0])
        #otherwise it is an input layer
        else:
            lay = layer.Layer(N, activation, 1)
