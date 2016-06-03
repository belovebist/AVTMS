"""
This is the main script that starts the program.
It handles all the initializations and starts-up
the program.
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

from modules import merge, objects


def main():
    """
    This is the starting part of the System
    000, 106, 211, 387, 510, 648, 702, 866, 1221
    """
    suffix = ['0106', '0211', '0387', '0510', '0648', '0702', '0866', '1221']

    # Load the reference views
    sourcedir = 'Images/multiple_views/'
    ref_views = [cv2.imread(sourcedir + 'view_' + str(i + 1) + '/view_' + str(i + 1) + '0000.png', cv2.IMREAD_COLOR) for i in range(4)]

    test_views = [[cv2.imread(sourcedir + 'view_' + str(i + 1) + '/view_' + str(i + 1) + x + '.png', cv2.IMREAD_COLOR) for i in range(4)] for x in suffix]

    # Making view_2 as reference view by passing view_2 as first element of list
    ref_views.append(ref_views.pop(0))

    for i in range(len(suffix)):
        test_views[i].append(test_views[i].pop(0))



    # Define a field to work on
    field = objects.Field(nCamera=4)
    field.initCameraParameters(ref_views)
    # Reconstruct the field's top view
    tv = field.reconstruct()

#    x = field.camera[0].getFrameDifference(test_views[0])

#    for i in range(4):
#        plt.subplot(2, 2, i + 1), plt.imshow(views[i])

    for i in range(len(suffix)):
        obj_pos = field.findObjectPosition(test_views[i])
        tv= merge.useOr([tv, cv2.merge([obj_pos, obj_pos, obj_pos])])


    plt.imshow(tv[:, :, -1::-1]), plt.title("Top View of the field")
#    plt.imshow(obj_pos, cmap='gray'), plt.title("Top View of the field")
    plt.show()


    pass

if __name__ == '__main__':
    main()
