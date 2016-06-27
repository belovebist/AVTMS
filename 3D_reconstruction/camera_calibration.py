"""
Camera calibration for 4 cameras
"""
import numpy as np
import cv2
import modules.calibrate as calibrate
import matplotlib.pyplot as plt


def imshow(img, *others):
    """ Displays in RGB format instead of BGR in case of color image """

    if len(img.shape) == 3:
        plt.imshow(img[:, :, -1::-1], *others)
    elif len(img.shape) == 2:
        plt.imshow(img, cmap='gray')


def main():
    src = 'Images/view1.png'

    img = cv2.imread(src, cv2.IMREAD_COLOR)
    size = img.shape

    input_corners = np.array(calibrate.extractCorners(calibrate.extractFieldLines(img)))
    output_corners = np.array([[0, 500.0, 0], [800.0, 500.0, 0], [800.0, 0, 0], [0, 0, 0]])

    input_corners = input_corners.astype('float32')
    output_corners = output_corners.astype('float32')

    cam_mat = cv2.initCameraMatrix2D([output_corners], [input_corners], size)
#    cam_mat = np.array([[2200, 0, 750], [0, 2200, 750.0], [0, 0, 1.0]])
#    dist_coefs = np.zeros(4)
#    projMat = cv2.calibrateCamera([output_corners], [input_corners], size, cam_mat, dist_coefs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)


    imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
