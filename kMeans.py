from random import randint
import numpy as np
from time import perf_counter as pf


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
def kMeans(NOofCentroids, data):
    data = np.array(data)
    dimensions = len(data[0])
    centroids = np.array([[randint(np.min(data[:, 0]), np.max(data[:, 0])) for _ in range(NOofCentroids)]]).transpose()#Initializes centroid with random values in the range of the data
    prevCentroids = np.array([])
    assignments = [[] for _ in range(NOofCentroids)]
    for dim in range(1, dimensions):
        centroids = np.append(centroids, np.array([[randint(np.min(data[:, dim]), np.max(data[:, dim])) for _ in range(NOofCentroids)]]).transpose(), 1)#Initializes centroid with random values in the range of the data
    while not np.array_equal(centroids, prevCentroids):#Loops until centroids do not change after an iteration
        temp = centroids
        assignments = [[] for _ in range(NOofCentroids)]
        for point in data:
            index = np.nanargmin([np.linalg.norm(point - centroid) for centroid in centroids])#Returns index of the centroid that is closest to point
            assignments[index].append(point)#Assigns each point to a centroid
        assignments = np.array(assignments)
        centroids = np.array([np.mean(assignments[index], axis = 0) for index in range(NOofCentroids)])#Gets the mean point of each assignments and makes it the new centroid for that cluster
        prevCentroids = temp
    assignments = assignments.tolist()
    for i in range(len(assignments)):
        if type(assignments[i]) == list:
            for j in range(len(assignments[i])):
                assignments[i][j] = assignments[i][j].tolist()
        else:
            assignments[i] = assignments[i].tolist()
    print(f'Centroids = {centroids.tolist()}; Assignments = {assignments}')
    return centroids.tolist(), assignments





data = [[1, 10, 100], [6, 30, 100], [9, 50, 300],]
print(kMeans(2, data))