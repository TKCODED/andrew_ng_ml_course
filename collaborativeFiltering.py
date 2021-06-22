import numpy as np
from time import perf_counter as pf
import random
from matplotlib import pyplot as plt


def main(data, NOfeatures):
    alpha = 0.000000001
    users = len(data[0])
    movies = len(data)
    data = np.array(data)  # Movie ratings by each user
    Theta = np.random.randint(1, 5, (
    NOfeatures + 1, users))  # Each column is how much one user prefers each genre: has a constant value
    features = [[random.random() for _ in range(NOfeatures)] for _ in range(movies)]
    for movie in features:
        movie.insert(0, 1)
    features = np.array(features)
    prevCost = 111111111111111111111
    costs = []
    iters = []
    cost = 111111111111
    print(Theta)
    print(features)
    index = 1
    while index < 10000:
        index += 1
        temp = cost
        for i in range(len(features)):  # NO of movies
            for j in range(len(features[0])):  # NO of features
                features[i][j] -= alpha * np.nansum(np.subtract(np.matmul(Theta.transpose()[i, :], features[i, :]), data[i][j]) * Theta[i][j])
        for i in range(len(features)):  # NO of movies
            for j in range(len(features[0])):  # NO of features
                Theta[i][j] -= alpha * np.nansum(np.subtract(np.matmul(Theta.transpose()[i, :], features[i, :]), data[i][j]) * features[i][j])
        cost = 0.5 * np.nansum(np.nansum(np.power(np.subtract(np.matmul(Theta.transpose(), features).transpose(), data), 2)))
        prevCost = temp
        iters.append(index)
        costs.append(cost)
    plt.plot(iters, costs)
    plt.show()
    print(cost)
    print(Theta)
    print(features)


data = [[1, 2, 3, 3], [0, 5, 4, 3], [5, 0, 0, 3]]  # Each row is how much one movie was liked by each user
main(data, 2)
