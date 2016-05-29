"""
This module contains methods that are used to detect particular lights from the input image
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


color_hsv = dict(
        red = [(np.array([0, 100, 100]), np.array([10, 255, 255])), 
            (np.array([150, 100, 100]), np.array([180, 255, 255]))], 
        yellow = [(np.array([25, 100, 100]), np.array([45, 255, 255]))], 
        green = [(np.array([40, 100, 100]), np.array([70, 255, 255]))]
        )


def getMask(hsv_img, color):
    """
    hsv_img = input image segment in hsv format
    color = color to be detected in HSV format including lower and upper bound i.e. color = (lower, upper)

    Returns the mask for image with given color
    """
    # Generate mask for all colors
    mask = 0
    for i in color:
        mask += cv2.inRange(hsv_img, i[0], i[1])

    return mask


def detectTrafficLight(img):
    """
    Gets an input image and returns the detected light
    """
    # Conversion from BGR to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red_mask = getMask(hsv_img, color_hsv['red'])
    yellow_mask = getMask(hsv_img, color_hsv['yellow'])
    green_mask = getMask(hsv_img, color_hsv['green'])

    mask = red_mask + yellow_mask + green_mask

    # Find the number of pixels for each color
    total = mask.size

    red = np.count_nonzero(red_mask)
    yellow = np.count_nonzero(yellow_mask)
    green = np.count_nonzero(green_mask)

    if red > yellow and red > green:
        return 'red', red, yellow, green
    elif yellow > red and yellow > green:
        return 'yellow', red, yellow, green
    elif green > red and green > yellow:
        return 'green', red, yellow, green
    else:
        return None



def main():

    img = cv2.imread('/home/bbist/Astha_Works/lightDetection/traffic1.jpeg', cv2.IMREAD_COLOR)

    print(detectTrafficLight(img))

    pass

if __name__ == '__main__':
    main()
