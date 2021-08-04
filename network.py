"""Initialize the Network class"""



import layer
import numpy as np

class Network:
    """Class used to initialize a multilayer perceptron"""
    def __init__(self, layers=None):
        """Attributes:
            Layers: A numpy array containing the layer objects
        """
        if layers == None: layers = np.array([])
        self.layers = layers

    def append(self, N, activation):
        """Append a layer of N neurons with specified activation to the network"""
        #check that there is a layer before
        if(self.layers.shape[0] != 0):
            lay = layer.Layer(N, activation, self.layers[-1].Neurons.shape[0])
        #otherwise it is an input layer
        else:
            lay = layer.Layer(N, activation, 1)
        self.layers = np.append(self.layers, lay)

    def forward(self, synapses):
        """Forward propegate through a network"""
        synapses = np.array([[i] for i in list(synapses)])
        for j in range(self.layers.shape[0]):
            print("Layer: " + str(j))
            new_synapses = np.array([])
            for i in range(self.layers[j].Neurons.shape[0]):
                new_synapse = self.layers[j].Neurons[i].forward(synapses[i])
                new_synapses = np.append(new_synapses, new_synapse)
            if j+1 < self.layers.shape[0]: new_synapses = np.array([list(new_synapses)] * self.layers[j+1].Neurons.shape[0]) 
            synapses = new_synapses
            print(new_synapses)
        return new_synapses
