"""
This module contains various methods for merging (fusing) multiple images
"""
import numpy as np


def useAverage(array):
    """
    Simply returns the average of N arrays
    """
    N = len(array)
    ret = array[0] / N
    for i in range(1, N):
        ret += array[i] / N

    return np.uint8(ret)


def useAnd(array):
    """
    Returns the pixelwise anding of multiple array
    """
    N = len(array)

    ret = array[0]
    for i in range(1, N):
        ret = ret & array[i]

    return ret


def useOr(array):
    """
    Returns the pixelwise anding of multiple array
    """
    N = len(array)

    ret = array[0]
    for i in range(1, N):
        ret |= array[i]

    return ret


def getWeightMatrix(wt, res, interpolation='linear'):
    """
    Returns the interpoleted weight matrix for given dimension

    wt : tuple containing weight for four corners
         in order (0, 0), (0, 1), (1, 1), (1, 0)
    """
    def getComponentWeight(wt, res):

        w, h = res
        mat = np.zeros(shape=(h, w), dtype=np.float)

        w, h = w - 1, h - 1

        for i in range(w):
            for j in range(h):
                mat[j][i] = wt * float(w - i)/w * float(h - j)/h

        return mat

    x0 = getComponentWeight(wt[0], res)
    x1 = getComponentWeight(wt[1], res)[-1::-1, :]
    x2 = getComponentWeight(wt[2], res)[-1::-1, -1::-1]
    x3 = getComponentWeight(wt[3], res)[:, -1::-1]

    return x0 + x1 + x2 + x3


def normalizeWeights(weight):
    """
    Returns the normalized weight
    """
    N = len(weight)
    x = weight[0]
    for i in range(1, N):
        x = x + weight[i]

    max = np.max(x)
    new_weight = [weight[i] / max for i in range(N)]

    return new_weight


def weightedAverage(array, weight):
    """
    Returns the weighted average of the input array
    """
    N = len(array)
    ret = np.zeros(shape=array[0].shape, dtype=np.float)
    for i in range(N):
        wt = np.dstack([weight[i], weight[i], weight[i]])
        ret += array[i] * wt

    return np.uint8(ret)
