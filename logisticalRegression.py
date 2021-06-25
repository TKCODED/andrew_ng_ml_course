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


def precproccessing(data):  # So infinities do not come from the const function
    data = np.subtract(data, np.mean(data, axis=0))
    data /= np.std(data, axis=0)
    data[:,0] = 1
    return data


def logisticalReg(inputs, outputs, RATE=1):
    NOofFeatures = len(inputs[0])
    NOofItems = len(inputs)
    mean = np.mean(np.array(inputs), axis=0)[1]#Gets the mean of the variabele
    std = np.std(np.array(inputs)[:,1])#Gets the standard deviation of the variabele
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
        temp = weights
        b = np.array(
            [1 / (1 + np.exp(-np.dot(temp.transpose(), inputs[i, :]))) for i in range(NOofItems)])

        weights = np.subtract(weights, ((RATE / NOofItems) * np.matmul(inputs.transpose(), np.array(
            [(1 / (1 + np.exp(-np.dot(temp.transpose(), inputs[i,:]))) - outputs[i]) for i in range(NOofItems)]))))
        activationOfInputs = [1 / (1 + np.exp(-np.dot(weights.transpose(), inputs[i,:]))) for i in range(NOofItems)]
        activationOfInputs = np.array([i.tolist() for i in activationOfInputs])
        activationOfInputs[activationOfInputs == 1] = 0.9999999
        activationOfInputs[activationOfInputs == 0] = 0.0000001
        cost = np.nansum(np.subtract(np.multiply(outputs, -np.log(activationOfInputs)),   np.multiply(np.subtract(1, outputs),
                                                 np.log(np.subtract(1, activationOfInputs))))) / NOofItems
        print(cost)
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
    print("Cost is:", cost)
    plt.plot(iters, costs)
    plt.show()
    return (weights)


inputs = [[1, i,i**2,i**3,i**4,i**5,i**6] for i in range(1,10)]
outputs = [1, 1, 1, 0, 0, 0, 0, 1, 1]
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

print(logisticalReg(inputs, outputs, RATE=1))

# print(logisticalReg(inputs, outputs))
