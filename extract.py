# Extracts the four corners of a rectangle

import cv2
import numpy as np

def getLines(img):
    """
    Extracts yellow lines and returns the image
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining range for yellow color in HSV
    lower_yellow = np.array([25, 40, 40])
    upper_yellow = np.array([45, 255, 255])

    # Threshold the HSV image to get only yellow color
    mask = cv2.inRange(hsv,lower_yellow,upper_yellow)

    # Mask the original image with the required color
    res = cv2.bitwise_and(img,img,mask=mask)

    return res
    # Conversion into grayscale image
    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    # Applying binary thresholding
    ret, thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY)

    return thresh

def main():
    imgg = cv2.imread('Images/multiple_views/view_1/view_10000.png')

    img_lines = getLines(imgg)

    img = cv2.cvtColor(img_lines, cv2.COLOR_BGR2GRAY)

    img_canny = cv2.Canny(img, 50, 50)

    img_blur = cv2.GaussianBlur(img_canny, (5, 5), 0)


    cv2.imshow('Image', imgg)
    cv2.waitKey(0)
    cv2.imshow('Image', img_lines)
    cv2.waitKey(0)
    cv2.imshow('Image', img_canny)
    cv2.waitKey(0)
    cv2.imshow('Image', img_blur)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    pass

if __name__ == '__main__':
    main()
