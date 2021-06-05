import numpy as np
from time import perf_counter as pf


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
    RATE = 0.00001
    NOofFeatures = len(inputs[0]) + 1
    NOofItems = len(inputs)
    for input in inputs:
        input.insert(0, 1)  # Inserts 1 into each group of label at index 0 to compensate for the constant weight.
    inputs = np.array(inputs, np.float64)
    weights = np.full((1, NOofFeatures), 10, np.float64)
    cost = 100
    prevCost = 88888888
    while cost != prevCost and cost != np.inf:
        tempCost = cost
        temp = weights
        for j in range(len(weights)):
            weights[j] -= (RATE / NOofItems) * sum(
                [(np.dot(temp, inputs[i]) - outputs[i]) * inputs[i] for i in range(NOofItems)])
        cost = np.nansum([outputs[i] * (-np.log(1 / (1 + np.exp(float(-(np.dot(weights, inputs[i]))))))) - (
                    1 - outputs[i]) * (np.log(1 - (1 / (1 + np.exp(float(-np.dot(weights, inputs[i]))))))) for i in
                              range(NOofItems)]) / NOofItems
        print(cost)
        prevCost = tempCost
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    string = str(weights[0][0])
    for index in range(1, len(weights[0])):
        string += ' +' + str(weights[0][index]) + '' + alphabet[index - 1]
    print(f"Function is: 1/1+e^-({string})")  # Prints Logistical Regression equation
    print("Cost is:", cost)
    return (weights)

inputs = [[2], [3], [4], [5], [6], [7]]
# for input in inputs:
#     input.insert(0, 1)
# inputs = np.array(inputs)
# weights = np.full((1, 2), 10)
# print([-np.log(1/(1 + np.exp(float(-np.dot(weights, inputs[i]))))) for i in range(len(inputs))])

outputs = [1, 1, 1, 0, 0, 0]
print(logisticalReg(inputs, outputs))
