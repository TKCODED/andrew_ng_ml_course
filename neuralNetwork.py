import numpy as np
import random
from time import perf_counter as pf
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


class network:
    def __init__(self, config, inputs, labels, lamda=0):
        self.config = config  # List of the number of units for each layer excluding the input layer and including the output layer
        self.input = self.inputLayer(inputs)
        self.layers = [self.layer(config[i]) for i in range(len(config))]
        self.labels = labels
        self.y = np.zeros((len(inputs), self.config[-1]))
        for index in range(len(
                self.labels)):  # Creates a matrix 0 that have 1's in the column of each correct label for every ith row
            self.y[index][self.labels[index] - 1] = 1
        self.results = []
        self.lamda = lamda
        self.cost = 0

    def forward(self, input):
        self.layers[0].activation = [self.sigmoid(np.dot(self.layers[0].weights[k, :], input)) for k in
                                     range(len(self.layers[0].weights))]
        self.layers[0].z = [np.dot(self.layers[0].weights[k, :], input) for k in
                            range(len(self.layers[0].weights))]
        self.layers[0].activation.insert(0, 1)
        self.layers[0].activation = np.array(self.layers[0].activation)
        for j in range(1, len(self.config) - 1):
            self.layers[j].activation = [
                self.sigmoid(np.dot(self.layers[j].weights[k, :], self.layers[j - 1].activation)) for k in
                range(len(self.layers[j].weights))]
            self.layers[j].activation.insert(0, 1)
            self.layers[j].activation = np.array(self.layers[j].activation)
        self.layers[-1].activation = [
            self.sigmoid(np.dot(self.layers[-1].weights[k, :], self.layers[-1 - 1].activation))
            for k in range(len(self.layers[-1].weights))]
        self.results.append(self.layers[-1].activation)

    def findCost(self):
        self.cost = -np.nansum(np.multiply(self.y[0:len(self.results), :], np.log(self.results)) - np.multiply(
            np.subtract(1, self.y[0:len(self.results), :]), np.log(self.results))) / len(self.results) + (
                                self.lamda / (2 * len(self.results))) * np.sum(
            np.sum([self.layers[j].weights for j in range(len(self.layers))]))

    def backward(self):
        dOut = np.subtract(self.layers[-1].activation, self.y[0:len(self.results), :])
        d = []
        d.append(dOut)
        deltas = []
        for j in range(len(self.layers) - 1):
            delta = np.multiply(d[j].transpose(), self.layers[-2 - j].activation.transpose())
            deltas.append(delta)
            d.append(np.multiply(d[-1], self.layers[-2 - j].weights[:, 1:-1]))
        deltas = deltas[::-1]
        print(deltas)
        weigthsGradients = np.divide(deltas, len(self.results))
        for j in range(1, len(self.layers)):
            np.add(self.layers[j].weights, weigthsGradients[j - 1])
        # TODO: number of weight gradients wrong the indexing is wrong

    def sigmoid(self, input):
        return 1 / (1 - np.exp(input))

    def sigmoidGradient(self, input):
        return sigmoid(input) * (1 - sigmoid(input))

    class layer:
        def __init__(self, units):
            self.units = units
            self.weights = np.array([[random.randint(0, 30) for _ in range(units + 1)] for _ in range(units)])
            self.activation = []
            self.z = []

    class inputLayer:
        def __init__(self, inputs):
            self.inputs = inputs
            for input in inputs:
                input.insert(0, 1)
            self.inputs = self.precproccessing(self.inputs)
            self.inputs[:, 0] = 1

        def precproccessing(self, data):  # So infinities do not come from the const function
            data = np.array(data)
            data = np.subtract(data, np.mean(data, axis=0))
            data /= np.ptp(data, axis=0)
            data *= 2
            return data


n = network([2, 2], [[3, 5], [4, 3]], [1, 2])
print([n.layers[j].weights for j in range(len(n.layers))])
n.forward(n.input.inputs[0])
n.findCost()
n.backward()
print([n.layers[j].weights for j in range(len(n.layers))])
