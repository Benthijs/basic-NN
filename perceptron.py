import numpy as np
import math




class Perceptron:
    """Represents a single neuron (perceptron) in a neural network"""
    def __init__(self, incoming, bias, activation):
        """Attributes:
            Incoming: A numpy array containing all weights
            Bias: An integer representing the bias
            Activation: A string denoting the desired activation function
        """
        self.incoming = incoming
        self.activation = activation
        self.bias = bias

    def forward(self, synapses):
        """The forward propegation in a perceptron"""
        print(self.incoming, self.activation)
        incoming = np.dot(self.incoming, synapses) + self.bias
        # interprets the activation string as a method 
        the_method = getattr(Perceptron, self.activation)
        print(incoming)
        out = the_method(self, incoming)
        return out

    def relu(self, incoming):
        if(incoming > 0):
            return incoming
        return 0

    def softmax(self,incoming):
        all_incoming = np.append(self.incoming, self.bias)
        print(all_incoming)
        y = math.e ** all_incoming
        tot = np.dot(np.ones(all_incoming.shape[0]), y)
        return (math.e ** incoming) / tot
