import numpy as np
import math




class Perceptron:
    """Represents a single neuron (perceptron) in a neural network"""
    def __init__(self, incoming, activation, outgoing):
        self.incoming = incoming
        self.activation = activation
        self.outgoing = outgoing

    def forward(self, synapses):
        """The forward propegation in a perceptron"""
        print(self.incoming, self.activation)
        incoming = self.incoming * synapses
        the_method = getattr(Perceptron, self.activation)
        out = the_method(self, incoming) * self.outgoing
        return out

    def softmax(self,incoming):
        y = math.e ** incoming
        tot = np.dot(np.ones(y.shape[-1]), y)
        return y / tot
