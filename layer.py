"""In this file we initialize the layer class"""
import numpy as np
import perceptron



class Layer:
    """Represents a layer of perceptrons"""
    def __init__(self, N, activation_function, n_before=1, n_after=1):
        """Attributes:
            Neurons: A numpy array containing all the perceptrons of the given layer
        In:
            n_before: Integer representing the number of neurons in the layer before
            n_after: Int representing # of neurons in proceeding layer
            N: int for # neurons in this layer
            activation_function: string representing activation function used
            """
        Neurons = []
        # randomly initialize each of the n perceptrons
        for _ in range(N):
            Neurons.append(perceptron.Perceptron(np.random.rand(n_before),activation_function,np.random.rand(n_after)))
        self.Neurons = np.array(Neurons)
