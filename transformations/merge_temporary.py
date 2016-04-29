"""
This script takes four perspective views of an image and merges them into one, and generates a approximated top view
"""


import cv2
from matplotlib import pyplot as plt
import numpy as np

import merge
import calibrate


def main():
    n_cols, n_rows = 2000, 1300


    sourcedir = '../Images/multiple_views/'
    views = [cv2.imread(sourcedir + 'view' + str(i + 1) + '.png', cv2.IMREAD_GRAYSCALE) for i in range(4)]


    corners = [np.float32([(13,101), (308, 531), (627, 219), (271, 9)]), 
            np.float32([(15, 247), (385, 547), (615, 98), (353, 17)]), 
            np.float32([(29, 92), (264, 494), (585, 238), (262, 18)]), 
            np.float32([(10, 285), (345, 503), (554, 81), (320, 8)])]
    
    proj_mat = calibrate.run(views, corners, res=(n_cols, n_rows))

    planar_views = [cv2.warpPerspective(views[i], proj_mat[i], (n_cols, n_rows)) for i in range(4)]
    
    simple_merged = merge.average(planar_views)



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
