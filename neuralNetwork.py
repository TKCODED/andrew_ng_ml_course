import numpy as np
from time import perf_counter as pf
import random
from matplotlib import pyplot as plt
import pandas as pd
import json
import os
from tensorflow.keras.datasets import mnist



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
    def __init__(self, config, inputs, labels):
        self.config = config  # List of the number of units for each layer excluding the input layer and including the output layer
        self.input = self.inputLayer(inputs)
        self.layers = [self.layer(config[i + 1], config[i]) for i in range(len(config) - 1)]
        self.layers.insert(0, self.layer(config[0], len(inputs[0])))
        self.labels = labels
        self.y = np.zeros((len(inputs), self.config[-1]))
        for index in range(len(self.labels)):  # Creates a 0 matrix that have 1's in the column of each correct label for every ith row
            self.y[index, self.labels[index][0]] = 1
        self.results = np.empty((0, config[-1]))
        self.cost = 0
        self.costs = []
        self.costsIters = []
        self.trainAccuracies = []
        self.trainAccuracyIters = []
        self.testAccuracies = []
        self.testAccuracyIters = []
        self.index = 0
        self.vdW = [np.zeros(layer.weights.shape) for layer in self.layers]
        self.sdW = [np.zeros(layer.weights.shape) for layer in self.layers]
        self.epsilon = 0.0000001
        self.bestWeights = []
        self.bestEpoch = 0

    def train(self, epochs=1, rate=1.0, lamda=0, batchSize=False, momentum=False, RMSprop=False, dropout=False, gradCheck=False):# Works only for balanced networks and regularization for gradients have not been added
        inputs = self.input.inputs
        if not batchSize:
            batchSize = len(inputs)
        self.costs = []
        self.costsIters = []
        self.trainAccuracies = []
        self.trainAccuracyIters = []
        self.testAccuracies = []
        self.testAccuracyIters = []
        for epoch in range(epochs):
            self.vdW = [np.zeros(layer.weights.shape) for layer in self.layers]#Reset vdW, why? as minibatch resets to 0 every epoch thus bias checking has to be reimplemented each epoch
            self.sdW = [np.zeros(layer.weights.shape) for layer in self.layers]
            for minibatch in range(len(inputs)//batchSize):
                #Forward
                batchInputs = inputs[batchSize*(minibatch):batchSize*(minibatch + 1), :]
                self.layers[0].z = np.matmul(batchInputs, self.layers[0].weights.transpose())
                self.layers[0].activation = self.ReLu(np.array(self.layers[0].z))
                # Dropout
                if dropout:  # Droput variable hold the percentage of units that are kept thus is from 0-1
                    dropMatrix = np.random.rand(self.layers[0].activation.shape[0],
                                                self.layers[0].activation.shape[1]) < dropout
                    self.layers[0].activation = np.multiply(self.layers[0].activation, dropMatrix)
                    self.layers[0].activation /= dropout
                    # Applied before putting in the bias term to not drop it
                if len(self.layers) - 1 != 0:
                    self.layers[0].activation = np.insert(self.layers[0].activation, 0, 1, axis=1)
                for l in range(1, len(self.layers) - 1):
                    self.layers[l].z= np.matmul(self.layers[l-1].activation, self.layers[l].weights.transpose())
                    self.layers[l].activation = self.ReLu(np.array(self.layers[l].z))
                    #Dropout
                    if dropout:  # Droput variable hold the percentage of units that are kept thus is from 0-1
                        dropMatrix = np.random.rand(self.layers[l].activation.shape[0], self.layers[l].activation.shape[1]) < dropout
                        self.layers[l].activation = np.multiply(self.layers[l].activation, dropMatrix)
                        self.layers[l].activation /= dropout
                        # Applied before putting in the bias term to not drop it
                    self.layers[l].activation = np.insert(self.layers[l].activation, 0, 1, axis=1)
                self.layers[-1].z = np.matmul(self.layers[-2].activation, self.layers[-1].weights.transpose())
                self.layers[-1].activation = self.sigmoid(np.array(self.layers[-1].z))
                self.results = np.vstack((self.results, self.layers[-1].activation))
                #Accuracy
                # results = np.equal(np.argmax(self.layers[-1].activation, axis=1), self.labels.transpose())
                # results = np.where(results == True, 1, results)
                # results = np.where(results == False, 0, results)
                # results = results[0]
                # trainAccuracy = np.sum(results) / len(results)
                # self.trainAccuracies.append(trainAccuracy)
                # self.trainAccuracyIters.append(self.index)
                # print(f"Epoch: {epoch + 1}  Minibatch: {minibatch + 1}")
                # print(f"Train Accuracy: {trainAccuracy * 100}%")
                # testAccuracy = self.test(x_test, y_test)
                # self.testAccuracies.append(testAccuracy)
                # self.testAccuracyIters.append(self.index)
                # print(f"Test Accuracy: {testAccuracy * 100}%")
                # if testAccuracy == max(self.testAccuracies):
                #     self.bestWeights = [layer.weights.tolist() for layer in self.layers]
                #     self.bestEpoch = epoch + 1

                #Backprop
                d = []
                y = self.y[batchSize*(minibatch):batchSize*(minibatch + 1), :]
                d.append(np.subtract(self.layers[-1].activation, y))#Does d for the output layer
                for l in range(len(self.layers) - 1):
                    epsilon = np.multiply(np.matmul(d[-1], self.layers[-1 - l].weights[:, 1:]), self.ReLuGradient(np.array(self.layers[-2 - l].z)))
                    d.append(epsilon)
                d = d[::-1]
                deltas =[]
                deltas.append(np.matmul(d[0].transpose(), batchInputs))# Does delta for batchInputs
                for l in range(1, len(self.layers)):
                    deltas.append(np.matmul(self.layers[l - 1].activation.transpose(), d[l]).transpose()/len(batchInputs))
                if not momentum and not RMSprop:
                    for j in range(len(self.layers)):
                        self.layers[j].weights = np.subtract(self.layers[j].weights, rate * np.add(deltas[j], (lamda/len(batchInputs))*self.layers[j].weights))
                elif momentum and not RMSprop:#Momentuum variable holds the beta value or the value that determines over how many previous gradients it averages
                    for j in range(len(self.layers)):
                        self.vdW[j] = np.add(momentum*self.vdW[j], (1-momentum)*deltas[j])
                        self.vdW[j] = np.divide(self.vdW[j], 1-(momentum**(minibatch+1)))
                        self.layers[j].weights = np.subtract(self.layers[j].weights, rate * np.add(self.vdW[j], (
                                    lamda / len(batchInputs)) * self.layers[j].weights))
                elif RMSprop and not momentum:#Momentuum variable holds the beta value or the value that determines over how many previous gradients it averages
                    for j in range(len(self.layers)):
                        self.sdW[j] = np.add(RMSprop*self.sdW[j], (1-RMSprop)*np.power(deltas[j], 2))
                        self.sdW[j] = np.divide(self.sdW[j], 1-(RMSprop**(minibatch+1)))
                        self.layers[j].weights = np.subtract(self.layers[j].weights, rate * np.add(np.divide(deltas[j], np.sqrt(self.sdW[j]) + self.epsilon), (
                                    lamda / len(batchInputs)) * self.layers[j].weights))
                elif momentum and RMSprop:#Momentuum variable holds the beta value or the value that determines over how many previous gradients it averages
                    for j in range(len(self.layers)):
                        self.vdW[j] = np.add(momentum * self.vdW[j], (1 - momentum) * deltas[j])
                        self.vdW[j] = np.divide(self.vdW[j], 1 - (momentum ** (minibatch + 1)))
                        self.sdW[j] = np.add(RMSprop * self.sdW[j], (1 - RMSprop) * np.power(deltas[j], 2))
                        self.sdW[j] = np.divide(self.sdW[j], 1 - (RMSprop ** (minibatch+1)))
                        self.layers[j].weights = np.subtract(self.layers[j].weights, rate * np.add(np.divide(self.vdW[j], np.sqrt(self.sdW[j]) + self.epsilon), (
                                    lamda / len(batchInputs)) * self.layers[j].weights))
                #GradCheck
                if gradCheck:#GradCheck variable holds epsilon for the amount a parameter is increased
                    dTheta = deltas[0].flatten()
                    for l in range(1, len(deltas)):
                        dTheta = np.hstack((dTheta, deltas[l].flatten()))
                    dThetaError = []
                    for l in range(len(self.layers)):#layers
                        for i in range(len(self.layers[l].weights)):#units
                            for k in range(len(self.layers.weights[i,:])):#weights
                                tempWeightsAdd = [layer.weights for layer in self.layers]
                                tempWeightsAdd[l][i][k] += gradCheck
                                tempWeightsSub = [layer.weights for layer in self.layers]
                                tempWeightsSub[l][i][k] -= gradCheck
                                activations = batchInputs
                                for weights in tempWeightsAdd:
                                    activations = self.ReLu(np.matmul(activations, weights.transpose()))
                                    activations[activations == 1] = 0.999999999999999999
                                    activations[activations == 0] = 0.000000000000000001
                                costAdded = -np.sum(np.multiply(self.y, activations) - np.multiply(np.subtract(1, self.y), np.log(np.subtract(1, activations)))) / len(activations) \
                                            + (lamda / (2 * len(activations))) * np.sum([np.sum(weights) for weights in tempWeightsAdd])
                                activations = batchInputs
                                for weights in tempWeightsSub:
                                    activations = self.ReLu(np.matmul(activations, weights.transpose()))
                                    activations[activations == 1] = 0.999999999999999999
                                    activations[activations == 0] = 0.000000000000000001
                                costSubbed = -np.sum(np.multiply(self.y, activations) - np.multiply(np.subtract(1, self.y), np.log(np.subtract(1, activations)))) / len(activations) \
                                            + (lamda / (2 * len(activations))) * np.sum([np.sum(weights) for weights in tempWeightsAdd])
                                error = (costAdded - costSubbed)/(2*gradCheck)
                                dThetaError.append(error)
                    dError = np.linalg.norm(dTheta, dThetaError)/(np.linalg.norm(dTheta) + np.linalg.norm(dThetaError))
                    print("Gradient Check Error:", dError)
            ###INSIDE LOOP##
            # Forward
            self.layers[0].z = np.matmul(inputs, self.layers[0].weights.transpose())
            self.layers[0].activation = self.ReLu(np.array(self.layers[0].z))
            # Dropout
            if dropout:  # Droput variable hold the percentage of units that are kept thus is from 0-1
                dropMatrix = np.random.rand(self.layers[0].activation.shape[0],
                                            self.layers[0].activation.shape[1]) < dropout
                self.layers[0].activation = np.multiply(self.layers[0].activation, dropMatrix)
                self.layers[0].activation /= dropout
                # Applied before putting in the bias term to not drop it
            if len(self.layers) - 1 != 0:
                self.layers[0].activation = np.insert(self.layers[0].activation, 0, 1, axis=1)
            for l in range(1, len(self.layers) - 1):
                self.layers[l].z = np.matmul(self.layers[l - 1].activation, self.layers[l].weights.transpose())
                self.layers[l].activation = self.ReLu(np.array(self.layers[l].z))
                # Dropout
                if dropout:  # Droput variable hold the percentage of units that are kept thus is from 0-1
                    dropMatrix = np.random.rand(self.layers[l].activation.shape[0],
                                                self.layers[l].activation.shape[1]) < dropout
                    self.layers[l].activation = np.multiply(self.layers[l].activation, dropMatrix)
                    self.layers[l].activation /= dropout
                    # Applied before putting in the bias term to not drop it
                self.layers[l].activation = np.insert(self.layers[l].activation, 0, 1, axis=1)
            self.layers[-1].z = np.matmul(self.layers[-2].activation, self.layers[-1].weights.transpose())
            self.layers[-1].activation = self.sigmoid(np.array(self.layers[-1].z))
            self.results = np.vstack((self.results, self.layers[-1].activation))
            # Accuracy
            results = np.equal(np.argmax(self.layers[-1].activation, axis=1), self.labels.transpose())
            results = np.where(results == True, 1, results)
            results = np.where(results == False, 0, results)
            results = results[0]
            trainAccuracy = np.sum(results) / len(results)
            self.trainAccuracies.append(trainAccuracy)
            self.trainAccuracyIters.append(self.index)
            print(f"Epoch: {epoch + 1}")
            print(f"Train Accuracy: {trainAccuracy * 100}%")
            testAccuracy = self.test(x_test, y_test)
            self.testAccuracies.append(testAccuracy)
            self.testAccuracyIters.append(self.index)
            print(f"Test Accuracy: {testAccuracy * 100}%")
            if testAccuracy == max(self.testAccuracies):
                self.bestWeights = [layer.weights.tolist() for layer in self.layers]
                self.bestEpoch = epoch + 1
            print("COST:", self.findCost(size=len(y), lamda=lamda))
            self.costs.append(self.findCost(size=len(y), lamda=lamda))
            self.costsIters.append(self.index)
            self.index += 1

    def findCost(self, size=None, lamda=0):
        if not size:
            size = len(self.y)# Finds cost after the entire dataset has been proccessed
        self.layers[-1].activation = np.round(self.layers[-1].activation, 10)
        self.layers[-1].activation[np.round(self.layers[-1].activation, 3) == 1] = 0.9999999999999
        self.layers[-1].activation[np.round(self.layers[-1].activation, 3) == 0] = 0.0000000000001
        self.cost = -np.sum(np.multiply(self.y[0:size,:], np.log(self.layers[-1].activation[0:size, :])) + np.multiply( #Takes the last size amount from self.results, which matches
            np.subtract(1, self.y[0:size, :]), np.log(np.subtract(1, self.layers[-1].activation[0:size, :])))) / len(self.layers[-1].activation[0:size, :]) \
             + (lamda / (2 * len(self.layers[-1].activation))) * np.sum([np.sum(self.layers[j].weights) for j in range(len(self.layers))])
        return self.cost

    def predict(self, inputs):
        inputs = np.array(inputs)
        inputs = self.preproccessing(inputs)
        for l in range(len(self.layers)):
            inputs = np.insert(inputs, 0, 1, axis=1)
            if l == 0:
                inputs = self.ReLu(np.matmul(inputs, self.layers[l].weights[:, 0:len(inputs)].transpose()))
            else:
                inputs = self.ReLu(np.matmul(inputs, self.layers[l].weights.transpose()))
        return inputs

    def test(self, inputs, outputs):
        outputs = np.array(outputs)
        predictions = np.argmax(self.predict(inputs), axis=1)
        results = np.equal(predictions, outputs.transpose())
        results = np.where(results == True, 1, results)
        results = np.where(results == False, 0, results)
        results = results[0]
        accuracy = np.sum(results) / len(results)
        return accuracy

    def plot(self, costs=True, trainAccuracies=True, testAccuracies=True):
        if costs:
            plt.plot(self.costsIters, self.costs)
            plt.xlabel("Training Iteration")
            plt.ylabel("Cost")
            plt.show()
        if trainAccuracies:
            plt.plot(self.trainAccuracyIters, self.trainAccuracies)
            plt.xlabel("Training Iteration")
            plt.ylabel("Training Accuracy")
            plt.show()
        if testAccuracies:
            plt.plot(self.testAccuracyIters, self.testAccuracies)
            plt.xlabel("Testing Iteration")
            plt.ylabel("Testing Accuracy")
            plt.show()

    def learningCurve(self, size=None): #Broken
        if not size:
            size = len(self.input.inputs)

        costs, iters = [], []
        for i in range(1, size):#Plots cost instead of accuracy
            self.train(self.input.inputs[0:i,:])#Even though data shape changes the shape of the weights dont.
            costs.append(self.findCost(size=i))
            iters.append(i + 1)
        plt.plot(iters, costs)
        plt.show()

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def sigmoidGradient(self, input):
        return np.multiply(self.sigmoid(input), (1 - self.sigmoid(input)))

    def ReLu(self, input):
        return np.maximum(input, 0)

    def ReLuGradient(self, input):
        input[input > 0] = 1
        input[input <= 0] = 0
        return input

    def softmax(self, input):
        input = np.exp(input)
        print(input.shape)
        print(np.sum(input, axis=1).reshape(-1,1).shape)
        input = np.divide(input, np.sum(input, axis=1).reshape(-1, 1))
        print(input.shape)
        return input

    def preproccessing(self, data):  # So infinities do not come from the const function
        data = np.subtract(data, np.mean(data, axis=0))
        std = np.std(data, axis=0)
        std[std == 0] = 1
        data = np.divide(data, std)
        return data

    def save(self, rate, batchSize=None, lamda=0):
        if not batchSize:
            batchSize = len(self.input.inputs)
        dirpath = os.path.dirname(__file__)
        modelPath = os.path.join(dirpath, f"neuralNetworkModels\\model-{self.config}-{batchSize}-{self.bestEpoch}-{'{0:.5f}'.format(rate)}-{lamda}-{'{0:.3f}'.format(max(self.trainAccuracies))}-{'{0:.3f}'.format(max(self.testAccuracies))}.json".replace(" ", ""))
        weights = {"weights": self.bestWeights}
        with open(modelPath, "w") as file:
            json.dump(weights, file)

    def load(self, filename):
        filename += '.json'
        dirpath = os.path.dirname(__file__)
        modelPath = os.path.join(dirpath, "neuralNetworkModels")
        modelPath = os.path.join(modelPath, filename)
        with open(modelPath, "r") as file:
            data = json.load(file)
            weights = data["weights"]
            for l in range(len(self.layers)):
                self.layers[l].weights = np.array(weights[l])

    class layer:
        def __init__(self, units, prevNOunits):
            self.units = units
            self.prevNOunits = prevNOunits
            self.weights = np.random.randn(units, prevNOunits + 1)*np.sqrt(2/prevNOunits)
            self.activation = []
            self.z = []

    class inputLayer:
        def __init__(self, inputs):
            self.inputs = inputs
            self.inputs = np.insert(self.inputs, 0, 1, axis=1)
            self.inputs = self.preproccessing(self.inputs)
            self.inputs[:, 0] = 1

        def preproccessing(self, data):  # So infinities do not come from the const function
            data = np.subtract(data, np.mean(data, axis=0))
            std = np.std(data, axis=0)
            std[std == 0] = 1
            data /= std
            return data




(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
y_train = y_train.reshape(-1,1)
# y_train = y_train[~np.isnan(x_train).any(axis=1)]#Removes rows from inputs and output of rows in input that have NAN values
# x_train = x_train[~np.isnan(x_train).any(axis=1)]
# x_train = x_train[~np.isnan(y_train).any(axis=1)]
# y_train = y_train[~np.isnan(y_train).any(axis=1)]#Removes rows from inputs and output of rows in input that have NAN values
x_test = x_test.reshape(10000, 784)
y_test = y_test.reshape(-1,1)
# y_test = y_test[~np.isnan(x_test).any(axis=1)]#Removes rows from inputs and output of rows in input that have NAN values
# x_test = x_test[~np.isnan(x_test).any(axis=1)]
# x_test = x_test[~np.isnan(y_test).any(axis=1)]
# y_test = y_test[~np.isnan(y_test).any(axis=1)]#Removes rows from inputs and output of rows in input that have NAN values
n = network([800, 10], x_train, y_train)
#n.load("model-[800,10]-3.000-0.000-0-0.726-0.723")
#print([layer.weights.shape for layer in n.layers])
n.train(epochs=100, rate=0.0025, batchSize=1000)#, momentum=0.9, RMSprop=0.999)
n.save(rate=0.0025, batchSize=1000)
n.plot()

#print([layer.weights for layer in n.layers])
