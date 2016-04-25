"""
This module contains various methods for merging (fusing) multiple images
"""

import numpy as np
import cv2


def simple_merge(*array, N = 1):
    """
    Simply returns the average of N arrays
    """
    ret = np.ndarray(shape=array[0].shape, dtype=np.uint8)
    for i in range(N):
        ret += array[i] / N

    return ret
