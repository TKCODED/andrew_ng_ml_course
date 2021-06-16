import numpy as np
from time import perf_counter as pf
import random
from matplotlib import pyplot as plt


def timer(function):
    def wrapper(*args, **kwargs):
        t0 = pf()
        proccessedFunc = function(*args, **kwargs)
        t1 = pf()
        time = t1 - t0
        print(function.__name__, "took", time, "second.")
        return proccessedFunc

    return wrapper


@timer
def logisticalReg(inputs, outputs):
    RATE = 0.0001
    NOofFeatures = len(inputs[0]) + 1
    NOofItems = len(inputs)
    for input in inputs:
        input.insert(0, 1)  # Inserts 1 into each group of label at index 0 to compensate for the constant weight.
    inputs = np.array(inputs, np.float64)
    outputs = np.array(outputs, np.float64).reshape(len(outputs), 1)
    weights =np.array([random.randint(1,3) for _ in range(NOofFeatures)], np.float64).reshape(NOofFeatures, 1)
    cost = 1000
    prevCost = 88888888
    index = 0
    iters = []
    costs = []
    value = 0
    minCost = 1000000
    while index < 10000:
        tempCost = cost
        temp = weights

        weights = np.subtract(weights,((RATE / NOofItems) * np.matmul(inputs.transpose(), np.array([1 / (1 + np.exp(float(-np.dot(temp.transpose(), inputs[i])))) - outputs[i] for i in range(NOofItems)]))))
        activationOfInputs = [1 / (1 + np.exp(float(-np.dot(weights.transpose(), inputs[i])))) for i in range(NOofItems)]
        activationOfInputsCleanedTemp = [0.00000000000000001 if x == 0 else x for x in activationOfInputs]
        activationOfInputsCleaned = [0.9999999999999999 if x == 1 else x for x in activationOfInputsCleanedTemp]
        cost = np.nansum(np.subtract(np.multiply(outputs, -np.log(activationOfInputsCleaned)), np.multiply(np.subtract(1, outputs), np.log(np.subtract(1, activationOfInputsCleaned))))) / NOofItems
        costs.append(cost)
        index += 1
        iters.append(index)
        prevCost = tempCost
        #print("Cost =",cost,"Weights =", weights)
        if cost < minCost:
            value = weights
            minCost = cost
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    string = str(weights[0])
    for index in range(1, len(weights)):
        string += ' +' + str(weights[index]) + '' + alphabet[index - 1]
    print(f"Function is: 1/1+e^-({string})")  # Prints Logistical Regression equation
    print("Cost is:", cost)
    plt.plot(iters, costs)
    plt.show()

    return (weights)

inputs = [[4,1], [9,4], [16,9], [25, 6], [36,25], [49,36], [64,49], [81,64], [100,81]]
# for input in inputs:
#     input.insert(0, 1)
# inputs = np.array(inputs)
# weights = np.full((1, 2), 10)
# print([-np.log(1/(1 + np.exp(float(-np.dot(weights, inputs[i]))))) for i in range(len(inputs))])

outputs = [0, 0, 0, 1, 1, 1, 1, 0, 0]
print(logisticalReg(inputs, outputs))
