"""A short example on how to use the perceptron class"""
import numpy as np
import perceptron
from random import random
network = perceptron.Perceptron(np.random.rand(3), random(), 'softmax')
print(network.forward(np.array([1,2,3])))

