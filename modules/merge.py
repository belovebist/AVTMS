"""
This module contains various methods for merging (fusing) multiple images
"""

import numpy as np
import cv2


def average(array):
    """
    Simply returns the average of N arrays
    """
    N = len(array)
    ret = np.ndarray(shape=array[0].shape, dtype=np.uint8)
    for i in range(N):
        ret += array[i] / N

    return ret

def weightedAverage(weight, array):
    """
    Returns the weighted average of the input array
    """
    N = len(array)
    ret = np.ndarray(shape=array[0].shape, dtype=np.uint8)
    for i in range(N):
        ret += array[i] * weight[i] / N

    return ret
    
