import numpy as np
import random
from time import perf_counter as pf
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


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
        self.layers = [self.layer(config[i]) for i in range(len(config))]
        self.labels = labels
        self.y = np.zeros((len(inputs), self.config[-1]))
        for index in range(len(
                self.labels)):  # Creates a 0 matrix that have 1's in the column of each correct label for every ith row
            self.y[index][self.labels[index] - 1] = 1
        self.results = np.empty((0, config[-1]))
        self.lamda = lamda
        self.cost = 0

    def forward(self, input):  # Takes in a datapoint at a time
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
            self.sigmoid(np.dot(self.layers[-1].weights[k, :], self.layers[-2].activation))
            for k in range(len(self.layers[-1].weights))]
        tempArr = np.array(self.layers[-1].activation).reshape(1, self.config[-1])
        self.results = np.vstack((self.results, tempArr[0]))
        self.results = np.round(self.results, 10)

    def findCost(self):  # Finds cost after the entire dataset has been proccessed
        self.results[self.results == 1] = 0.999999999
        self.results[self.results == 0] = 0.000000001
        self.cost = -np.sum(np.multiply(self.y[0:len(self.results), :], np.log(self.results)) - np.multiply(
            np.subtract(1, self.y[0:len(self.results), :]), np.log(self.results))) / len(self.results) + (
                            self.lamda / (2 * len(self.results))) * np.sum(
            np.sum([self.layers[j].weights for j in range(len(self.layers))]))
        return self.cost

    def backward(self, input):  # Takes in a datapoint at a time
        dOut = np.subtract(self.y[len(self.results) - 1, :], self.layers[
            -1].activation)  # It uses the length of results to get the index of the input to get the correct y-value
        dOut = dOut.reshape(-1, 1)
        d = []
        d.append(dOut)
        deltas = []
        for j in range(len(self.layers) - 1):
            # print(d[j].shape, j)
            # print(self.layers[-2 - j].activation.reshape(-1,1).transpose().shape, j)
            delta = np.matmul(d[j], self.layers[-2 - j].activation.reshape(-1, 1).transpose())
            deltas.append(delta)
            sigmoidGradient = self.sigmoidGradient(np.array(self.layers[-2 - j].z)).reshape(-1, 1)
            d.append(np.multiply(np.matmul(self.layers[-2 - j].weights[:, 1:], d[-1]),
                                 sigmoidGradient))  # SIGMOID GRADIENT INDEX MIGHT BE WRONG
        delta = np.multiply(d[-1], input.reshape(-1, 1).transpose())
        deltas.append(delta)
        # d.append(np.multiply(d[-1], self.layers[-2 - j].weights[:, 1:-1])) d1 not needed as we are not going to change the inputs of the network
        deltas = deltas[::-1]
        weightsGradients = deltas
        return weightsGradients
        # for j in range(1, len(self.layers)):
        #     np.add(self.layers[j].weights, weigthsGradients[j - 1])
        # TODO: number of weight gradients wrong the indexing is wrong
        # TODO: Weights Gradients divide by the length of self.results which does not make sense since the backprop algorithm only does it for one input

    def batch(self):
        self.forward(self.input.inputs[0])
        weightsGradients = self.backward(self.input.inputs[0])
        for i in range(1, len(self.input.inputs)):
            print("--------------------------")
            self.forward(self.input.inputs[i])
            weightsGradients = np.add(weightsGradients, self.backward(self.input.inputs[i]))
        for j in range(len(self.layers)):
            self.layers[j].weights = np.add(self.layers[j].weights, weightsGradients[j])
        print("COST:", self.findCost())

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def sigmoidGradient(self, input):
        return np.multiply(self.sigmoid(input), (1 - self.sigmoid(input)))

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
            data /= np.std(data, axis=0)
            return data


n = network([2, 2], [[3, 5], [4, 3]], [1, 2])
print("WEIGHTS:", [n.layers[j].weights for j in range(len(n.layers))])
n.batch()
print("WEIGHTS:", [n.layers[j].weights for j in range(len(n.layers))])
