"""
This is the main script that starts the program.
It handles all the initializations and starts-up
the program.
"""
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

from modules import merge, objects


def main():
    """
    This is the starting part of the System
    000, 106, 211, 387, 510, 648, 702, 866, 1221
    """

    srcDir = os.getcwd() + '/Images/multiple_views'

    channels = [srcDir + '/view_2/view_2',
                srcDir + '/view_3/view_3',
                srcDir + '/view_4/view_4',
                srcDir + '/view_1/view_1']

    # Define a field to work on
    field = objects.Field(nCamera=4)

    field.initCameraChannels(channels)
    field.initCameraParameters()

    field.reconstruct()

    tv = field.startTracking()


    plt.imshow(tv[:, :, -1::-1]), plt.title("Top View of the field")
    plt.show()


    pass

if __name__ == '__main__':
    main()
