import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from modules import calibrate


def imshow(img, *others):
    """ Displays in RGB format instead of BGR in case of color image """

    if len(img.shape) == 3:
        plt.imshow(img[:, :, -1::-1], *others)
    elif len(img.shape) == 2:
        plt.imshow(img, cmap='gray')

def showCameraParameters(arg):

    ret, mtx, dist, rvecs, tvecs = arg

    print('\nReturn Value : \n', ret)
    print('\nCamera Matrix : \n', mtx)
    print('\nDistortion Coefficients : \n', dist)
    print('\nRotation Vector : \n', rvecs)
    print('\nTranslation Vector : \n', tvecs)


def main():
    """
    Finds the calibration parameters
    """
    src = '../Images'
    img_ref = cv2.imread('/'.join([src, 'view1.png']), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    img = calibrate.extractFieldLines(img_ref)
    corners = calibrate.extractCorners(img)

    scale = (0.65, 1)
    obj_pts = np.array([[0, 0, 0], [0, scale[1], 0], [scale[0], scale[1], 0], [scale[0], 0, 0]], dtype=np.float32)
    img_pts = np.array([x for x in corners], dtype=np.float32)

    print('Object Points :\n', obj_pts)
    print('\nImage Points :\n', img_pts)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([obj_pts], [img_pts], gray.shape[::-1], None, None)

    showCameraParameters((ret, mtx, dist, rvecs, tvecs))

#    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img.shape, 1, img.shape)
#    dst = cv2.undistort(img_ref, mtx, dist, None, mtx)
#    x, y, w, h = roi
#    dst = dst[y:y+h, x:x+w]

#    cv2.line(img_ref, (int(corners[0][0]), int(corners[0][1])), (int(corners[1][0]), int(corners[1][1])), (0, 0, 255), 1)
#    rvecs, tvecs, inliers = cv2.solvePnPRansac(obj_pts, img_pts, mtx, dist)

#    newCameraMtx = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    obj_pts = np.array([[0, 0.8, 0], [0.5, 0.5, 0], [0, 1, 0], [.4, .5, 0]])
    imgPts, *b = cv2.projectPoints(obj_pts, rvecs[0], tvecs[0], mtx, dist)

    imgPts = np.int64(imgPts)
    cv2.circle(img_ref, (imgPts[0][0][0], imgPts[0][0][1]), 1, (0, 0, 255), -1)
    cv2.circle(img_ref, (imgPts[1][0][0], imgPts[1][0][1]), 1, (0, 0, 255), -1)
    cv2.circle(img_ref, (imgPts[2][0][0], imgPts[2][0][1]), 1, (0, 0, 255), -1)
    cv2.circle(img_ref, (imgPts[3][0][0], imgPts[3][0][1]), 1, (0, 0, 255), -1)
#    imshow(np.hstack([img_ref, dst]))
    imshow(img_ref)
    plt.show()


if __name__ == '__main__':
    main()
