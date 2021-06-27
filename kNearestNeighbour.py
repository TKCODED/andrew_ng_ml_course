from random import randrange
import numpy as np
from time import perf_counter as pf
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def kNearestsNeighnours(k, data, input):
    values = np.array(list(data.values()))
    values = values.reshape(values.shape[0]*values.shape[1], values.shape[2]).tolist()
    nearest = []
    groups = [0]*k
    print(values)
    for value in values:
        if len(nearest) < k:
            nearest.append(value)
        elif np.linalg.norm(np.array(value) - np.array(input)) < np.linalg.norm(np.array(nearest[-1]) - np.array(input)):
            nearest[-1] = value
        nearest = sorted(nearest)
    for point in nearest:
        for group in range(len(list(data.values()))):
            if input in list(data.values())[group]:
                groups[group] += 1
    group = np.argmax(groups)
    return group


dic = {0:[[1,2,3],[3,5,6]], 1:[[7,8,9],[10,11,12]]}
print(list(dic.values()))
print(kNearestsNeighnours(3, dic, [3,5,7]))