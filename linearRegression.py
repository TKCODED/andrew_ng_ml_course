import numpy as np
from time import perf_counter as pf
from matplotlib import pyplot as plt


def timer(function):
    def wrapper(*args, **kwargs):
        t0 = pf()
        proccessedFunc = function(*args, **kwargs)
        t1 = pf()
        time = t1 - t0
        print(function.__name__,"took",time,"second.")
        return proccessedFunc
    return wrapper


@timer
def statsLinReg(points) -> tuple:
    """y = a + bx"""
    xAVG = sum([point[0] for point in points]) / len(points)
    SSXX = sum([(xAVG - point[0]) ** 2 for point in points])
    xDIFF = [(xAVG - point[0]) for point in points]
    yAVG = sum([point[1] for point in points]) / len(points)
    yDIFF = [(yAVG - point[1]) for point in points]
    SSXY = sum([xDIFF[index] * yDIFF[index] for index in range(len(xDIFF))])
    b = SSXY / SSXX
    a = yAVG - b * xAVG
    print(f'y = {a} + {b}x')
    return a, b


@timer
def linReg(inputs, outputs) -> tuple:
    RATE = 0.001
    constant = 10
    gradient = 10
    cost = (1 / 2 * len(inputs)) * sum([(constant + gradient * inputs[i] - outputs[i]) ** 2 for i in range(len(inputs))])
    prevCost = 88888888
    while cost != prevCost:
        temp1, temp2, temp3 = gradient, constant, cost
        constant -= (RATE / len(inputs)) * sum([(temp2 + temp1*inputs[i] - outputs[i]) for i in range(len(inputs))])
        gradient -= ((RATE / len(inputs)) * sum([(temp2 + temp1*inputs[i] - outputs[i])*inputs[i] for i in range(len(inputs))]))
        cost = (1/2*len(inputs))* sum([(constant + gradient*inputs[i] - outputs[i])**2 for i in range(len(inputs))])
        prevCost = temp3

    print(f'{round(constant, 2)} + {round(gradient, 2)}x')
    return constant, gradient


@timer
def multiVarLinReg(inputs, outputs) -> np.ndarray:
    """"
    Takes in 2D-array. Each array is one grouping of each labels values(one point on n-dimensional plane.
    """
    RATE = 0.001
    NOofFeatures = len(inputs[0]) + 1
    NOofItems = len(inputs)
    for input in inputs:
        input.insert(0, 1)#Inserts 1 into each group of label at index 0 to compensate for the constant weight.
    inputs = np.array(inputs)
    weights = np.full((1, NOofFeatures), 10, np.float32)
    cost = 100
    prevCost = 88888888
    index = 0
    iter = []
    costs = []
    while cost != prevCost:
        tempCost = cost
        temp = weights
        for j in range(len(weights)):
            weights[j] -= (RATE/NOofItems)*sum([(np.dot(temp, inputs[i]) - outputs[i])*inputs[i] for i in range(NOofItems)])
        cost = float(sum([(np.dot(weights, inputs[i]) - outputs[i])**2 for i in range(NOofItems)]))/(2*NOofItems)
        costs.append(cost)
        index += 1
        iter.append(index)
        prevCost = tempCost
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    string = str(weights[0][0])
    for index in range(1, len(weights[0])):
        string +=' +' + str(weights[0][index]) + '' + alphabet[index - 1]
    print(string)#Prints Linear Regression equation
    print("Cost is:", cost)
    print(iter, costs)
    plt.plot(iter, cost)
    #plt.show()
    return(weights)


input = [[3, 4, 7, 8, 9], [11, 12, 15, 16, 17], [13, 14, 17, 18, 19], [20, 21, 24, 25, 26]]
output = [50, 60, 70, 80]  # First value is constant and thus needs one more value than the number of inputs
multiVarLinReg(input, output)
#linReg([10,20,30],[20,40,60])
#statsLinReg([[10,20], [30,40], [50,60]])
