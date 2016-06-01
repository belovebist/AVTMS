"""
This script takes four perspective views of an image and merges them into one, and generates a approximated top view
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

from modules import merge, calibrate


def main():
    n_cols, n_rows = 1000, 550

    sourcedir = 'Images/multiple_views/'

    views = [cv2.imread(sourcedir + 'view_' + str(i + 1) + '/view_' + str(i + 1) + '0000.png', cv2.IMREAD_COLOR) for i in range(4)]

    # Making view_2 as reference view by passing view_2 as first element of list
    views.append(views.pop(0))

    _views, proj_mat = calibrate.run(views, res=(n_cols, n_rows))

    planar_views = [cv2.warpPerspective(views[i], proj_mat[i], (n_cols, n_rows)) for i in range(4)]

    simple_merged = merge.useAverage(planar_views)



#    for i in range(4):
#        plt.subplot(3, 4, 2 * i + 1), plt.imshow(views[i], cmap='gray'), plt.title('Input_' + str(i + 1))
#        plt.subplot(3, 4, 2 * i + 2), plt.imshow(planar_views[i], cmap='gray'), plt.title('View_' + str(i + 1))

#    plt.subplot(3, 4, 9),
    plt.imshow(simple_merged, cmap='gray'), plt.title('Merged View')
#    plt.subplot(1, 2, 1), plt.imshow(simple_merged, cmap = 'gray'), plt.title('Merged_view')

#    plt.subplot(1, 2, 2), plt.imshow(opposite_merged, cmap = 'gray'), plt.title('Merged_view')

    plt.show()

    pass

if __name__ == '__main__':
    main()
