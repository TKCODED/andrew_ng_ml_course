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


def pca(data, dimensions=None, variance=0.99):  # PCA takes in a bunch of points and maps it onto a given dimension
    data = np.array(data)
    cov = np.cov(data.transpose())
    U, S, V = np.linalg.svd(cov)
    if not dimensions:
        for k in reversed(range(len(data[0]))): #Picks the kth dimension, which gives minumum variance above the variance value
            if sum(S[0:k])/sum(S) < variance:
                dimensions = k+1
    Udim = U[:, 0:dimensions]
    z = np.array([np.matmul(Udim.transpose(), data[i, :]) for i in range(len(data))])
    dataApprox = np.array([np.matmul(Udim, z[i]) for i in range(len(z))])
    return z, dataApprox


def visualize(initial, reduced):
    ax = plt.axes(projection="3d")
    ax.scatter3D(initial[:,0], initial[:,1], initial[:,2])
    plt.show()

    fig = plt.figure()
    plt.grid('on')
    ax = fig.gca()
    ax.scatter(reduced[:,0], reduced[:,1])
    plt.show()



data = [[0, 0, 0,0], [2, 6, 8, 10], [10, 15, 20, 25], [30, 40, 50, 60]]
a, b = pca(precproccessing(data), dimensions=2)
visualize(b, a)
