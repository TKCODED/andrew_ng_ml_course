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
        print(function.__name__, "took", time, "second.")
        return proccessedFunc
    return wrapper


def epsilonInput(epsilons=None):
    if epsilons is None:
        epsilons = [1 / (n + 1) for n in range(10)]

    def decorator(function):
        @timer
        def wrapper(*args, **kwargs):
            bestEpsilon = 0
            bestAccuracy = 0
            for epsilon in epsilons:
                result = function(epsilon=epsilon, *args, **kwargs)
                if result > bestAccuracy:
                    bestEpsilon = epsilon
                    bestAccuracy = result
            return bestEpsilon, bestAccuracy

        return wrapper

    return decorator


def inList(nparray, list):
    for npelement in list:
        if np.array_equal(nparray, npelement):
            return True
    return False


def multiVariateGaussian(data, labels=None, epsilon=0.5, x=None, graph=True):  # labels are 1 for anomalous data or 0 for normal data
    data = np.array(data)
    covar = np.cov(data.transpose())
    mean = np.mean(data, axis=0)
    anomalous = []
    normal = []
    for point in data:
        normPoint = np.subtract(point, mean)
        p = (1 / ((2 * np.pi) ** (len(data[0]) / 2) * (np.linalg.det(covar) ** 0.5))) * np.exp(
            -0.5 * (np.matmul(np.matmul(normPoint.transpose(), np.linalg.pinv(covar)), normPoint)))
        anomalous.append(point) if p < epsilon else normal.append(point)
    incorrect = 0
    for index in range(len(labels) if labels else 0):
        if labels[index] == 1 and inList(data[index], normal):
            incorrect += 1
        elif labels[index] == 0 and inList(data[index], anomalous):
            incorrect += 1
    if labels:
        # print("Accuracy is:", 1 - (incorrect / len(data)))
        pass
    if x:
        normPoint = np.subtract(x, mean)
        p = (1 / ((2 * np.pi ** (len(data[0]) / 2)) * (np.linalg.det(covar) ** 0.5))) * np.exp(
            -0.5 * (np.matmul(np.matmul(normPoint.transpose(), np.linalg.pinv(covar)), normPoint)))
        print("Input is anomalous.") if p < epsilon else print("Input is normal.")
    if len(data[0]) == 3 and graph:
        ax = plt.axes(projection='3d')
        if normal:
            ax.scatter3D(np.array(normal)[:, 0], np.array(normal)[:, 1], np.array(normal)[:, 2], label='Normal')
        if anomalous:
            ax.scatter3D(np.array(anomalous)[:, 0], np.array(anomalous)[:, 1], np.array(anomalous)[:, 2],
                         label='Anomalies')
        if x:
            ax.scatter3D(x[0], x[1], x[2], label='Input')
        ax.legend()
        plt.show()
    if len(data[0]) == 2 and graph:
        fig = plt.figure()
        ax = fig.gca()
        if normal:
            ax.scatter(np.array(normal)[:, 0], np.array(normal)[:, 1], label='Normal')
        if anomalous:
            ax.scatter(np.array(anomalous)[:, 0], np.array(anomalous)[:, 1], label='Anomalous')
        if x: ax.scatter()
        plt.show(x[0], x[1], label='Input')
    if labels:
        return 1 - (incorrect / len(data))


def multiProductGaussian(data, labels=None, epsilon=0.5, x=None, graph=True):
    data = np.array(data)
    mean = np.mean(data, axis=0)
    var =  np.var(data, axis = 0)
    anomalous = []
    normal = []
    for point in data:
        p = np.prod([(1/((2*np.pi*var[feature])**0.5)) * np.exp(-((point[feature] - mean[feature])**2)/(2*var[feature])) for feature in range(len(data[0]))])
        anomalous.append(point) if p < epsilon else normal.append(point)
    incorrect = 0
    for index in range(len(labels) if labels else 0):
        if labels[index] == 1 and inList(data[index], normal):
            incorrect += 1
        elif labels[index] == 0 and inList(data[index], anomalous):
            incorrect += 1
    if labels:
        # print("Accuracy is:", 1 - (incorrect / len(data)))
        pass
    if x:
        p = np.prod([(1/((2*np.pi*var[feature])**0.5)) * np.exp(-((x[feature] - mean[feature])**2)/(2*var[feature])) for feature in range(len(data[0]))])
        print("Input is anomalous.") if p < epsilon else print("Input is normal.")
    if len(data[0]) == 3 and graph:
        ax = plt.axes(projection='3d')
        if normal:
            ax.scatter3D(np.array(normal)[:, 0], np.array(normal)[:, 1], np.array(normal)[:, 2], label='Normal')
        if anomalous:
            ax.scatter3D(np.array(anomalous)[:, 0], np.array(anomalous)[:, 1], np.array(anomalous)[:, 2],
                         label='Anomalies')
        if x:
            ax.scatter3D(x[0], x[1], x[2], label='Input')
        ax.legend()
        plt.show()
    if len(data[0]) == 2 and graph:
        fig = plt.figure()
        ax = fig.gca()
        if normal:
            ax.scatter(np.array(normal)[:, 0], np.array(normal)[:, 1], label='Normal')
        if anomalous:
            ax.scatter(np.array(anomalous)[:, 0], np.array(anomalous)[:, 1], label='Anomalous')
        if x: ax.scatter()
        plt.show(x[0], x[1], label='Input')
    if labels:
        return 1 - (incorrect / len(data))


data = [[1, 1, 1], [5, 5, 5], [5, 6, 5], [4, 5, 6], [7, 5, 6], [8, 8, 8], [5, 4, 4], [9, 1, 3], [5, 7, 5], [5, 6, 5],
        [4, 4, 6], [5.5, 5, 5], [5, 5.5, 5], [4.5, 5, 5.5]]
labelz = [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]

#If u want to see the graph of the most accurate epsilon set graph to True and epsilons list to only that value
print(epsilonInput(epsilons=[1 / (10 + (n / 2)) for n in range(1, 10000)])(multiProductGaussian)(data, labels=labelz, graph=False))
print(epsilonInput(epsilons=[1 / (10 + (n / 2)) for n in range(1, 10000)])(multiVariateGaussian)(data, labels=labelz, graph=False))
