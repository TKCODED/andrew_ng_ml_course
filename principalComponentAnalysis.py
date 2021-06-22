import numpy as np
from time import perf_counter as pf
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def precproccessing(data):  # So that after PCA each feature is relatively same to each other so the graph is not huge
    data = np.array(data)
    data = np.subtract(data, np.mean(data, axis=0))
    data /= np.ptp(data, axis=0)
    data *= 2
    return data


def pca(data, dimensions):  # PCA takes in a bunch of points and maps it onto a given dimension
    data = np.array(data)
    cov = np.cov(data.transpose())
    U, S, V = np.linalg.svd(cov)
    Udim = U[:, 0:dimensions]
    z = np.array([np.matmul(Udim.transpose(), data[i, :]) for i in range(len(data))])
    dataApprox = np.array([np.matmul(Udim, z[i]) for i in range(len(z))])
    return z, dataApprox


data = [[0, 0, 0], [2, 6, 8]]
print(pca(data, 1))
