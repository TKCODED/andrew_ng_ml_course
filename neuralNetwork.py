import numpy as np
from time import perf_counter as pf
import random
from matplotlib import pyplot as plt
import pandas as pd


def timer(function):
    def wrapper(*args, **kwargs):
        t0 = pf()
        proccessedFunc = function(*args, **kwargs)
        t1 = pf()
        time = t1 - t0
        print(function.__name__, "took", time, "second.")
        return proccessedFunc
    return wrapper


class network:
    def __init__(self, config, inputs, labels, lamda=0):
        self.config = config  # List of the number of units for each layer excluding the input layer and including the output layer
        self.input = self.inputLayer(inputs)
        self.layers = [self.layer(config[i + 1], config[i]) for i in range(len(config) - 1)]
        self.layers.insert(0, self.layer(config[0], len(inputs[0]) - 1))
        self.labels = labels
        self.y = np.zeros((len(inputs), self.config[-1]))
        for index in range(len(
                self.labels)):  # Creates a 0 matrix that have 1's in the column of each correct label for every ith row
            self.y[index][self.labels[index] - 1] = 1
        self.results = np.empty((0, config[-1]))
        self.lamda = lamda
        self.cost = 0

    def train(self, inputs):# Works only for balanced networks and regularization for gradients have not been added
        #Forward
        self.layers[0].z = np.matmul(inputs, self.layers[0].weights.transpose())
        self.layers[0].activation = self.sigmoid(self.layers[0].z)
        self.layers[0].activation = np.insert(self.layers[0].activation, 0, 1, axis=1)
        for l in range(1, len(self.layers)):
            self.layers[l].z= np.matmul(self.layers[l-1].activation, self.layers[l].weights.transpose())
            self.layers[l].activation = self.sigmoid(self.layers[l].z)
            if l != len(self.layers) - 1:
                self.layers[l].activation = np.insert(self.layers[l].activation, 0, 1, axis=1)
        self.results = np.vstack((self.results, self.layers[-1].activation))
       #Backprop
        d = []
        d.append(np.subtract(self.layers[-1].activation, self.y))#Does d for the output layer
        for l in range(len(self.layers) - 1):
            epsilon = np.multiply(np.matmul(d[-1], self.layers[-2 - l].weights[:, 1:]), self.sigmoidGradient(self.layers[-2 - l].z))
            d.append(epsilon)
        d = d[::-1]
        deltas =[]
        deltas.append(np.matmul(d[0].transpose(), inputs))# Does delta for inputs
        for l in range(1, len(self.layers)):
            deltas.append(np.matmul(self.layers[len(self.layers) - 1 - l].activation.transpose(), d[l]).transpose()/len(inputs))
        for j in range(len(self.layers)):
            self.layers[j].weights = np.add(self.layers[j].weights, deltas[j])
        print("COST:", self.findCost())

    def findCost(self):  # Finds cost after the entire dataset has been proccessed
        self.results = np.round(self.results, 10)
        self.results[self.results == 1] = 0.999999999
        self.results[self.results == 0] = 0.000000001
        self.cost = -np.sum(np.multiply(self.y[0:len(self.results), :], np.log(self.results)) - np.multiply(
            np.subtract(1, self.y[0:len(self.results), :]), np.log(self.results))) / len(self.results) \
                    #+ (self.lamda / (2 * len(self.results))) * np.sum(np.sum([self.layers[j].weights for j in range(len(self.layers))]))
        return self.cost

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def sigmoidGradient(self, input):
        return np.multiply(self.sigmoid(input), (1 - self.sigmoid(input)))

    class layer:
        def __init__(self, units, prevNOunits):
            self.units = units
            self.prevNOunits = prevNOunits
            self.weights = np.array([[random.randint(0, 30) for _ in range(self.units + 1)] for _ in range(self.prevNOunits)])
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
            data /= np.std(data, axis=0)
            return data

n = network([3, 3], [[3, 5, 3], [4, 3, 7]], [3, 1])
print("WEIGHTS:", [n.layers[j].weights for j in range(len(n.layers))])
n.train(n.input.inputs)
print("WEIGHTS:", [n.layers[j].weights for j in range(len(n.layers))])