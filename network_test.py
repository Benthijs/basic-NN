"""Initialize unit test for network class"""




import network
import perceptron
import numpy as np

network = network.Network()
network.append(5, 'relu')
network.append(10, 'relu')
network.append(1, 'softmax')
print(network.forward(np.random.rand(5)))
