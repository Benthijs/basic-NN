"""Initializes the model class"""




import network

class Model:
    """Class used to train the multilayer perceptron"""
    def __init__(self, network, loss='mse', optimizer='gradient-descent'):
        """network: A network object representing the multilayer perceptron
            loss: A string representing the used loss function
            optimizer: A string representing the algorithm used to perform learning"""
        self.network = network
        self.loss = loss
        self.optimizer = optimizer

    def train(self, epochs, data):
        """Function used to train the network
            epochs: Integer representing number of times to perform training
            data: An ndarray of data used to train the network"""
        for i in range(epochs):
            print("Epoch {}/{}".format(i, epochs))
            # perform forwards prop with data
            # calculate errors using loss and data
            # perform backpropagation
            # update network using optimizer

