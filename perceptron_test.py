"""A short example on how to use the perceptron class"""
import numpy as np
import perceptron
network = perceptron.Perceptron(np.random.rand(3), 'softmax', np.random.rand(3))
print(network.forward(np.array([1,2,3])))

