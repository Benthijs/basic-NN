import numpy as np
import math




class Perceptron:
    """Represents a single neuron (perceptron) in a neural network"""
    def __init__(self, incoming, bias, activation):
        self.incoming = incoming
        self.activation = activation
        self.bias = bias

    def forward(self, synapses):
        """The forward propegation in a perceptron"""
        print(self.incoming, self.activation)
        incoming = np.dot(self.incoming, synapses) + self.bias
        the_method = getattr(Perceptron, self.activation)
        out = the_method(self, incoming)
        return out

    def relu(self, incoming):
        if(incoming > 0):
            return incoming
        return 0

    def softmax(self,incoming):
        y = math.e ** self.incoming
        tot = np.dot(np.ones(self.incoming.shape[0]), y)
        return (math.e ** incoming) / tot
