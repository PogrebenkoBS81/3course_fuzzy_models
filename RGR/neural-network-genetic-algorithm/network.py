  
"""Class that represents the network to be evolved."""
import random
import logging

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class Network():
    """Represent a network and let us operate on it.
    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network."""

        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters.
        self.clf = None

    def create_random(self):
        """Create a random network."""

        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.
        Args:
            network (dict): The network parameters
        """

        self.network = network

    def train(self, x, y):
        """Train the network and record the accuracy.
        Args:
            x (DataFrame): df of all collected data params
            y (DataFrame): df of same height with normal/abnormal classification of x
        """

        # If model already trained - skip it.
        if self.accuracy != 0.:
            return
        # Create MLP classifier with given params.
        self.clf = MLPClassifier(**self.network)
        # Get datasets for training/testing, then train and test model.
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)
        self.clf.fit(x_train, y_train)
        y_pred = self.clf.predict(x_test)
        # set curr model accuracy
        self.accuracy = accuracy_score(y_test, y_pred)

    def print_network(self):
        """Print out a network."""

        print(self.network)
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))