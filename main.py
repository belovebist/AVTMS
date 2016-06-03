"""
This is the main script that starts the program.
It handles all the initializations and starts-up
the program.
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

import modules.objects as obj


def main():
    """
    This is the starting part of the System
    000, 106, 211, 387, 510, 648, 702, 866, 1221
    """
    # Load the reference views
    sourcedir = 'Images/multiple_views/'
    ref_views = [cv2.imread(sourcedir + 'view_' + str(i + 1) + '/view_' + str(i + 1) + '0000.png', cv2.IMREAD_COLOR) for i in range(4)]

    test_views = [cv2.imread(sourcedir + 'view_' + str(i + 1) + '/view_' + str(i + 1) + '1221.png', cv2.IMREAD_COLOR) for i in range(4)]

    # Making view_2 as reference view by passing view_2 as first element of list
    ref_views.append(ref_views.pop(0))
    test_views.append(test_views.pop(0))



    # Define a field to work on
    field = obj.Field(nCamera=4)
    field.initCameraParameters(ref_views)
    # Reconstruct the field's top view
    tv = field.reconstruct()

    obj_pos = field.findObjectPosition(test_views)
#    x = field.camera[0].getFrameDifference(test_views[0])

#    plt.imshow(tv[:, :, -1::-1]), plt.title("Top View of the field")
    plt.imshow(obj_pos, cmap='gray'), plt.title("Top View of the field")
    plt.show()


    pass

if __name__ == '__main__':
    main()
