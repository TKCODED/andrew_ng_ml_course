from random import randrange
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


@timer
def kMeans(NOofCentroids, data):
    data = np.array(data)
    dimensions = len(data[0])
    centroids = np.array([[randrange(np.min(np.ceil(data[:, 0])), np.max(np.ceil(data[:, 0]))) for _ in range(
        NOofCentroids)]]).transpose()  # Initializes centroid with random values in the range of the data
    prevCentroids = np.array([])
    assignments = [[] for _ in range(NOofCentroids)]
    for dim in range(1, dimensions):
        centroids = np.append(centroids, np.array(
            [[randrange(np.min(np.ceil(data[:, dim])), np.max(np.ceil(data[:, dim]))) for _ in
              range(NOofCentroids)]]).transpose(),
                              1)  # Initializes centroid with random values in the range of the data
    while not np.array_equal(centroids, prevCentroids):  # Loops until centroids do not change after an iteration
        temp = centroids
        assignments = [[] for _ in range(NOofCentroids)]
        for point in data:
            index = np.nanargmin([np.linalg.norm(point - centroid) for centroid in
                                  centroids])  # Returns index of the centroid that is closest to point
            assignments[index].append(point)  # Assigns each point to a centroid
        assignments = np.array(assignments)
        centroids = np.array([np.mean(assignments[index], axis=0) for index in range(
            NOofCentroids)])  # Gets the mean point of each assignments and makes it the new centroid for that cluster
        prevCentroids = temp
    assignments = assignments.tolist()
    if dimensions == 2:
        for i in range(len(assignments)):
            if type(assignments[i]) == list:
                for j in range(len(assignments[i])):
                    assignments[i][j] = assignments[i][j].tolist()
            else:
                assignments[i] = assignments[i].tolist()
    if dimensions == 2:
        fig = plt.figure()
        ax = fig.gca()
        for assignment in assignments:
            ax.scatter(np.array(assignment)[:, 0], np.array(assignment)[:, 1])
        for centroid in centroids:
            ax.scatter(centroid[0], centroid[1])
        plt.show()
    if dimensions == 3:

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for assignment in assignments:
            ax.scatter3D(np.array(assignment)[:, 0], np.array(assignment)[:, 1], np.array(assignment)[:, 2])
        for centroid in centroids:
            ax.scatter3D(centroid[0], centroid[1], centroid[2])
        plt.show()

    string = f'Centroids = {centroids.tolist()}; Assignments ='
    for assignment in assignments:
        string += f'{assignment}////////////////////////////'
    print(string)
    return centroids.tolist(), assignments


# Anderson Iris Set; Features included: Sepal Length and Width, Petal Length
'''
data = [[4.9, 3.0]
    , [4.7, 3.2]
    , [4.6, 3.1]
    , [5.0, 3.6]
    , [5.4, 3.9]
    , [4.6, 3.4]
    , [5.0, 3.4]
    , [4.4, 2.9]
    , [4.9, 3.1]
    , [5.4, 3.7]
    , [4.8, 3.4]
    , [4.8, 3.0]
    , [4.3, 3.0]
    , [5.8, 4.0]
    , [5.7, 4.4]
    , [5.4, 3.9]
    , [5.1, 3.5]
    , [5.7, 3.8]
    , [5.1, 3.8]
    , [5.4, 3.4]
    , [5.1, 3.7]
    , [4.6, 3.6]
    , [5.1, 3.3]
    , [4.8, 3.4]
    , [5.0, 3.0]
    , [5.0, 3.4]
    , [5.2, 3.5]
    , [5.2, 3.4]
    , [4.7, 3.2]
    , [4.8, 3.1]
    , [5.4, 3.4]
    , [5.2, 4.1]
    , [5.5, 4.2]
    , [4.9, 3.1]
    , [5.0, 3.2]
    , [5.5, 3.5]
    , [4.9, 3.6]
    , [4.4, 3.0]
    , [5.1, 3.4]
    , [5.0, 3.5]
    , [4.5, 2.3]
    , [4.4, 3.2]
    , [5.0, 3.5]
    , [5.1, 3.8]
    , [4.8, 3.0]
    , [5.1, 3.8]
    , [4.6, 3.2]
    , [5.3, 3.7]
    , [5.0, 3.3]
    , [7.0, 3.2]
    , [6.4, 3.2]
    , [6.9, 3.1]
    , [5.5, 2.3]
    , [6.5, 2.8]
    , [5.7, 2.8]
    , [6.3, 3.3]
    , [4.9, 2.4]
    , [6.6, 2.9]
    , [5.2, 2.7]
    , [5.0, 2.0]
    , [5.9, 3.0]
    , [6.0, 2.2]
    , [6.1, 2.9]
    , [5.6, 2.9]
    , [6.7, 3.1]
    , [5.6, 3.0]
    , [5.8, 2.7]
    , [6.2, 2.2]
    , [5.6, 2.5]
    , [5.9, 3.2]
    , [6.1, 2.8]
    , [6.3, 2.5]
    , [6.1, 2.8]
    , [6.4, 2.9]
    , [6.6, 3.0]
    , [6.8, 2.8]
    , [6.7, 3.0]
    , [6.0, 2.9]
    , [5.7, 2.6]
    , [5.5, 2.4]
    , [5.5, 2.4]
    , [5.8, 2.7]
    , [6.0, 2.7]
    , [5.4, 3.0]
    , [6.0, 3.4]
    , [6.7, 3.1]
    , [6.3, 2.3]
    , [5.6, 3.0]
    , [5.5, 2.5]
    , [5.5, 2.6]
    , [6.1, 3.0]
    , [5.8, 2.6]
    , [5.0, 2.3]
    , [5.6, 2.7]
    , [5.7, 3.0]
    , [5.7, 2.9]
    , [6.2, 2.9]
    , [5.1, 2.5]]
'''
data = [[5.1, 3.5, 1.4]
    , [4.9, 3.0, 1.4]
    , [4.7, 3.2, 1.3]
    , [4.6, 3.1, 1.5]
    , [5.0, 3.6, 1.4]
    , [5.4, 3.9, 1.7]
    , [4.6, 3.4, 1.4]
    , [5.0, 3.4, 1.5]
    , [4.4, 2.9, 1.4]
    , [4.9, 3.1, 1.5]
    , [5.4, 3.7, 1.5]
    , [4.8, 3.4, 1.6]
    , [4.8, 3.0, 1.4]
    , [4.3, 3.0, 1.1]
    , [5.8, 4.0, 1.2]
    , [5.7, 4.4, 1.5]
    , [5.4, 3.9, 1.3]
    , [5.1, 3.5, 1.4]
    , [5.7, 3.8, 1.7]
    , [5.1, 3.8, 1.5]
    , [5.4, 3.4, 1.7]
    , [5.1, 3.7, 1.5]
    , [4.6, 3.6, 1.0]
    , [5.1, 3.3, 1.7]
    , [4.8, 3.4, 1.9]
    , [5.0, 3.0, 1.6]
    , [5.0, 3.4, 1.6]
    , [5.2, 3.5, 1.5]
    , [5.2, 3.4, 1.4]
    , [4.7, 3.2, 1.6]
    , [4.8, 3.1, 1.6]
    , [5.4, 3.4, 1.5]
    , [5.2, 4.1, 1.5]
    , [5.5, 4.2, 1.4]
    , [4.9, 3.1, 1.5]
    , [5.0, 3.2, 1.2]
    , [5.5, 3.5, 1.3]
    , [4.9, 3.6, 1.4]
    , [4.4, 3.0, 1.3]
    , [5.1, 3.4, 1.5]
    , [5.0, 3.5, 1.3]
    , [4.5, 2.3, 1.3]
    , [4.4, 3.2, 1.3]
    , [5.0, 3.5, 1.6]
    , [5.1, 3.8, 1.9]
    , [4.8, 3.0, 1.4]
    , [5.1, 3.8, 1.6]
    , [4.6, 3.2, 1.4]
    , [5.3, 3.7, 1.5]
    , [5.0, 3.3, 1.4]
    , [7.0, 3.2, 4.7]
    , [6.4, 3.2, 4.5]
    , [6.9, 3.1, 4.9]
    , [5.5, 2.3, 4.0]
    , [6.5, 2.8, 4.6]
    , [5.7, 2.8, 4.5]
    , [6.3, 3.3, 4.7]
    , [4.9, 2.4, 3.3]
    , [6.6, 2.9, 4.6]
    , [5.2, 2.7, 3.9]
    , [5.0, 2.0, 3.5]
    , [5.9, 3.0, 4.2]
    , [6.0, 2.2, 4.0]
    , [6.1, 2.9, 4.7]
    , [5.6, 2.9, 3.6]
    , [6.7, 3.1, 4.4]
    , [5.6, 3.0, 4.5]
    , [5.8, 2.7, 4.1]
    , [6.2, 2.2, 4.5]
    , [5.6, 2.5, 3.9]
    , [5.9, 3.2, 4.8]
    , [6.1, 2.8, 4.0]
    , [6.3, 2.5, 4.9]
    , [6.1, 2.8, 4.7]
    , [6.4, 2.9, 4.3]
    , [6.6, 3.0, 4.4]
    , [6.8, 2.8, 4.8]
    , [6.7, 3.0, 5.0]
    , [6.0, 2.9, 4.5]
    , [5.7, 2.6, 3.5]
    , [5.5, 2.4, 3.8]
    , [5.5, 2.4, 3.7]
    , [5.8, 2.7, 3.9]
    , [6.0, 2.7, 5.1]
    , [5.4, 3.0, 4.5]
    , [6.0, 3.4, 4.5]
    , [6.7, 3.1, 4.7]
    , [6.3, 2.3, 4.4]
    , [5.6, 3.0, 4.1]
    , [5.5, 2.5, 4.0]
    , [5.5, 2.6, 4.4]
    , [6.1, 3.0, 4.6]
    , [5.8, 2.6, 4.0]
    , [5.0, 2.3, 3.3]
    , [5.6, 2.7, 4.2]
    , [5.7, 3.0, 4.2]
    , [5.7, 2.9, 4.2]
    , [6.2, 2.9, 4.3]
    , [5.1, 2.5, 3.0]
    , [5.7, 2.8, 4.1]]

kMeans(2, data)
