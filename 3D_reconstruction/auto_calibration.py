"""
Auto-calibration script for camera using live input feed
"""
import cv2
import numpy as np
import pickle


def getFrame(cam, flipped=True):
    ret, frame = cam.read()

    return frame if not flipped else cv2.flip(frame, 1)

def showCameraParameters(arg):

    ret, mtx, dist, rvecs, tvecs = arg

    print('\nReturn Value : \n', ret)
    print('\nCamera Matrix : \n', mtx)
    print('\nDistortion Coefficients : \n', dist)
    print('\nRotation Vector : \n', rvecs)
    print('\nTranslation Vector : \n', tvecs)

def calibrate(camera_port, grid_size = (7, 6), img_count = 12, skip_frames = 15):
    """
    This routine takes the live input from the specified camera port and finds
    the intrinsic and extrinsic camera parameters using opencv functions. This
    method requires a chessboard to be shown continuously from different angles
    in front of the camera to be calibrated. It automatically takes images few
    frames apart and use them to find camera parameters

    Arguments :
        camera_port  : port number of camera to be calibrated
        grid_size    : size of the grid to be searched for in chessboard
        img_count    : maximum number of images to use for calibration
        skip_frams   : minimum number of frames to be skipped during calibration

    """
    cam = cv2.VideoCapture(camera_port)

    # termination criteria for the calibration process
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points for given grid size
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

    # array to store object points and image points from all the images
    obj_points = []
    img_points = []

    # input images during calibration
    images = []

    # run the continuous loop to capture images and then calibrate
    skip_count = 0
    count = 0

    while count < img_count:
        ret, img = cam.read()
        img = cv2.flip(img, 1)

        # display the image
        cv2.imshow('Calibration', img)
        cv2.waitKey(1)

        if ret is False:
            continue
        skip_count += 1
        if skip_count < skip_frames:
            continue
        skip_count = 0

        # find the chessboard corners
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        # if found, add object points, image points (after refining them)
        if ret is True:
            print("Image", count, "Captured.")
            images.append(gray)
            count += 1
            obj_points.append(objp)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

            # draw and display the corners
            cv2.drawChessboardCorners(img, grid_size, corners, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # get the re-projection error
    mean_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i],
                img_points2, cv2.NORM_L2) / len(img_points2)
        mean_error += error

    print("Mean error : ", mean_error / len(obj_points))

    return (ret, mtx, dist, rvecs, tvecs), images


def main():

    # default port number of integrated webcam
    camera_port = 0

    # start calibration
    ret, images = calibrate(camera_port)

    # load the parameter values
#    ret = ret_val, mtx, dist, rvecs, tvecs = pickle.load(
#            open("camera_param.prm", "rb"))

    ret_val, mtx, dist, rvecs, tvecs = ret
    showCameraParameters(ret)

    # open a file to save the data
    camera_param = open("camera_param.prm", "wb")
    pickle.dump(ret, camera_param)


if __name__ == '__main__':
    main()
