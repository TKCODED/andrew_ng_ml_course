import numpy as np
from time import perf_counter as pf
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


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
    rate = 0.001
    constant = 10
    gradient = 10
    cost = (1 / 2 * len(inputs)) * sum([(constant + gradient * inputs[i] - outputs[i]) ** 2 for i in range(len(inputs))])
    prevCost = 88888888
    while cost != prevCost:
        temp1, temp2, temp3 = gradient, constant, cost
        constant -= (rate / len(inputs)) * sum([(temp2 + temp1*inputs[i] - outputs[i]) for i in range(len(inputs))])
        gradient -= ((rate / len(inputs)) * sum([(temp2 + temp1*inputs[i] - outputs[i])*inputs[i] for i in range(len(inputs))]))
        cost = (1/2*len(inputs))* sum([(constant + gradient*inputs[i] - outputs[i])**2 for i in range(len(inputs))])
        prevCost = temp3

    print(f'y = {round(constant, 2)} + {round(gradient, 2)}x')
    print("COST:", cost)
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(inputs, outputs)
    x = np.linspace(np.amin(np.array(inputs)) - 5, np.amax(np.array(inputs)) + 5, num=1000)
    y = constant + gradient*x
    plt.plot(x, y, 'r', label=f'y = {round(constant, 2)} + {round(gradient, 2)}x')
    plt.title(f'Graph of y = {round(constant, 2)} + {round(gradient, 2)}x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    return constant, gradient


@timer
def multiVarLinReg(inputs, outputs, rate=1) -> np.ndarray:
    """"
    Takes in 2D-array. Each array is one grouping of each labels values(one point on n-dimensional plane.
    """
    NOofFeatures = len(inputs[0]) + 1
    NOofItems = len(inputs)
    for input in inputs:
        input.insert(0, 1)#Inserts 1 into each group of label at index 0 to compensate for the constant weight.
    inputs = np.array(inputs)
    weights = np.full((1, NOofFeatures), 10, np.float32)
    cost = float(sum([(np.dot(weights, inputs[i]) - outputs[i])**2 for i in range(NOofItems)]))/(2*NOofItems)
    prevCost = 88888888
    index = 0
    iter = []
    costs = []
    while cost < prevCost:
        tempCost = cost
        temp = weights
        for j in range(len(weights)):
            weights[j] -= (rate/NOofItems)*np.sum([(np.matmul(temp, inputs[i]) - outputs[i])*inputs[i] for i in range(NOofItems)])
        cost = np.sum([(np.matmul(weights, inputs[i]) - outputs[i])**2 for i in range(NOofItems)])/(2*NOofItems)
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
    plt.plot(iter, costs)
    plt.show()

    ax = plt.axes(projection='3d')
    points = inputs.tolist()
    for i in range(len(points)):
        points[i].append(outputs[i])
    points = np.array(points)
    ax.scatter3D(points[:, 1], points[:, 2], points[:, 3])
    x = np.linspace(np.amin(points[:, 1], axis=-1) - 5, np.amax(points[:, 1], axis=-1) + 5, num=1000)
    y = np.linspace(np.amin(points[:, 2], axis=-1) - 5, np.amax(points[:, 2], axis=-1) + 5, num=1000)
    line = np.array([weights[0][0] + weights[0][1]*x[i] + weights[0][2]*y[i] for i in range(1000)])
    ax.plot3D(x, y, line, color='r', label=f'Graph of z = {round(weights[0][0], 2)} + {round(weights[0][1], 2)}x + {round(weights[0][2], 2)}y')
    plt.title(f'Graph of z = {round(weights[0][0], 2)} + {round(weights[0][1], 2)}x + {round(weights[0][2], 2)}y')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    return(weights)


input = [[3,4], [11, 12], [13, 14], [20, 21]]
output = [50, 60, 70, 80]  # First value is constant and thus needs one more value than the number of inputs
multiVarLinReg(input, output, rate=0.001)
#linReg([10,20,30],[20,40,60])
#statsLinReg([[10,20], [30,40], [50,60]])
