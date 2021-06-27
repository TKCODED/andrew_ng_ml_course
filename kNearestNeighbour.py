from random import randrange
import numpy as np
from time import perf_counter as pf
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def kNearestsNeighnours(k, data, input):
    dimensions = len(input)
    values = np.array(list(data.values()))
    values = np.concatenate(values, 0).tolist()
    nearest = []
    groups = [0] * len(data.keys())
    if len(values) < k:
        print("k is bigger than dataset!")
        return None
    for value in values:
        if len(nearest) < k:
            nearest.append(value)
        elif np.linalg.norm(np.array(value) - np.array(input)) < np.linalg.norm(
                np.array(nearest[-1]) - np.array(input)):
            nearest[-1] = value
        nearest = sorted(nearest, key=lambda point: float(np.linalg.norm(np.array(point) - np.array(input))))
    for point in nearest:
        for group in range(len(list(data.values()))):
            if point in list(data.values())[group]:
                groups[group] += 1
    group = np.argmax(groups)
    if dimensions == 2:
        colors = 'rbgcmy'
        fig = plt.figure()
        ax = fig.gca()
        for index in range(len(data.values())):
            ax.scatter(np.array(np.array(list(data.values()))[index])[:, 0],
                         np.array(np.array(list(data.values()))[index])[:, 1], c=colors[index])
            if group == index:
                ax.scatter(input[0], input[1], c=colors[index], marker='^')
    if dimensions == 3:
        colors = 'rbgcmy'
        ax = plt.axes(projection='3d')
        for index in range(len(data.values())):
            ax.scatter3D(np.array(np.array(list(data.values()))[index])[:, 0],
                       np.array(np.array(list(data.values()))[index])[:, 1],
                       np.array(np.array(list(data.values()))[index])[:, 2], c=colors[index])
            if group == index:
                ax.scatter3D(input[0], input[1], input[2], c=colors[index], marker='^')
    plt.show()


dic = {0: [[1, 2,3], [4, 5, 6]], 1: [[2, 5, 2], [3, 4, 1], [7, 8, 6], [10, 11, 7]]}
print(kNearestsNeighnours(3, dic, [3, 5, 5]))
