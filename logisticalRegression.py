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


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def precproccessing(data):  # So infinities do not come from the const function
    data = np.subtract(data, np.mean(data, axis=0))
    data /= np.std(data, axis=0)
    data[:, 0] = 1
    return data


def logisticalReg(inputs, outputs, RATE=1):
    NOofFeatures = len(inputs[0])
    NOofItems = len(inputs)
    mean = np.mean(np.array(inputs), axis=0)[1]  # Gets the mean of the variabele
    std = np.std(np.array(inputs)[:, 1])  # Gets the standard deviation of the variabele
    inputs = precproccessing(inputs)
    outputs = np.array(outputs, np.float64).reshape(len(outputs), 1)
    weights = np.array([random.randint(1, 3) for _ in range(NOofFeatures)], np.float64).reshape(NOofFeatures, 1)
    cost = 1000
    prevCost = 88888888
    index = 0
    iters = []
    costs = []

    while index < 50000:
        tempCost = cost
        weights = np.subtract(weights, ((RATE / NOofItems) * np.matmul(inputs.transpose(), sigmoid(np.matmul(inputs ,weights)) - outputs)))
        activationOfInputs = sigmoid(np.matmul(inputs ,weights)) - outputs
        #activationOfInputs = np.array([i.tolist() for i in activationOfInputs])
        activationOfInputs[activationOfInputs == 1] = 0.9999999
        activationOfInputs[activationOfInputs == 0] = 0.0000001
        cost = np.nansum(
            np.subtract(np.multiply(outputs, -np.log(activationOfInputs)), np.multiply(np.subtract(1, outputs),
                                                                                       np.log(np.subtract(1,
                                                                                                          activationOfInputs))))) / NOofItems
        costs.append(cost)
        index += 1
        iters.append(index)
        prevCost = tempCost
        if cost > prevCost:
            break
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    string = str(weights[0])
    for index in range(1, len(weights)):
        string += ' +' + str(weights[index]) + '' + alphabet[index - 1]
    print(f'Mean = {mean}, Standard Deviation = {std}')
    print(f"Function is: 1/1+e^-({string})")  # Prints Logistical Regression equation
    print("Cost is:", prevCost)
    x = np.linspace(0, len(inputs), 1000)
    y = np.zeros(1000)
    expression = [weights[i] * (((x - mean) / std) ** i) for i in range(len(weights))]
    for term in expression:
        y = np.add(y, term)
    plt.xlim([0, len(outputs) + 1])
    plt.ylim([-10, 10])
    plt.grid()
    plt.title("Decision Boundary")
    points = []
    for point in range(len(outputs)):
        points.append(point + 1) if outputs[point] == 1 else None
    plt.scatter(points, [0 for _ in range(len(points))])
    points = []
    for point in range(len(outputs)):
        points.append(point + 1) if outputs[point] == 0 else None
    plt.scatter(points, [0 for _ in range(len(points))], marker='^')
    plt.plot(x, y, 'r')
    # plt.plot(iters, costs)
    plt.show()
    return (weights, cost)


def learningCurve(inputs, outputs, rate=1, size=None):
    if not size:
        size = len(inputs)
    costs, iters = [], []
    for i in range(1, size):
        cost = logisticalReg(inputs[0:i], outputs[0:i], RATE=rate)[1]
        costs.append(cost)
        iters.append(i)
    plt.plot(iters, costs)
    plt.show()


outputs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
inputs = [[1, i, i**2, i**3] for i in range(1, len(outputs) + 1)]
# df = pd.read_csv('bottle.csv')
# input = df['Salnty'].to_numpy()
# inputs = np.array([np.power(input,i) for i in range(3)]).transpose()#Power to the 0 means the x1 column is included
# outputs = df['T_degC'].to_numpy()
# outputs = output.reshape(len(outputs), 1)
# outputs = output[~np.isnan(inputs).any(axis=1)]#Removes rows from inputs and output of rows in input that have NAN values
# inputs = inputs[~np.isnan(inputs).any(axis=1)]
# inputs = inputs[~np.isnan(outputs).any(axis=1)]
# outputs = outputs[~np.isnan(outputs).any(axis=1)]#Removes rows from inputs and output of rows in input that have NAN values
# outputs[outputs <= 15] = 0
# outputs[outputs > 15] = 1

print(logisticalReg(inputs, outputs, RATE=0.1))

# print(logisticalReg(inputs, outputs))
