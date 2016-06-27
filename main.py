"""
This is the main script that starts the program.
It handles all the initializations and starts-up
the program.
"""
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from modules import objects


def main():
    """
    This is the starting part of the System
    """

    srcDir = os.getcwd() + '/Images/multiple_views'

    channels = [srcDir + '/view_1/view_1',
                srcDir + '/view_2/view_2',
                srcDir + '/view_3/view_3',
                srcDir + '/view_4/view_4']

    # Define a field to work on
    field = objects.Field(nCamera=4)

    field.initCameraChannels(channels)
    field.initCameraParameters()

    field.reconstruct(mergeType='weighted', updateTV=True)

    tv = field.startTracking()
    """
    view = []
    for i in range(field.nCamera):
        view.append(cv2.warpPerspective(field.camera[i].refView, field.camera[i].projMatrix, field.res))

    for i in range(field.nCamera):
        plt.subplot(2, 2, i + 1), plt.imshow(view[i][:, :, -1::-1]), plt.title("Projected View " + str(i))
    """
    plt.imshow(tv[:, :, -1::-1]), plt.title("Top View of the Field")
    plt.show()

if __name__ == '__main__':
    main()
