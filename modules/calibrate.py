"""
This module contains camera calibration routines
The final result is the calibrated camera matrices that
perspectively transforms from image plane to common projection plane
(i.e. Top view)
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np


def extractFieldLines(img):
    """
    This method masks the field and extracts the lines of required color
    """
    # To convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining range for yellow color in HSV
    lower_yellow = np.array([25, 40, 40])
    upper_yellow = np.array([45, 255, 255])

    # Threshold the HSV image to get only yellow color
    mask = cv2.inRange(hsv,lower_yellow,upper_yellow)

    # Mask the original image with the required color
    res = cv2.bitwise_and(img,img,mask=mask)

    # Conversion into grayscale image
    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    # Applying binary thresholding
    ret, thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('image', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return thresh

def extractCorners(img):
    """
    Extracts the corner points of the rectangular field by solving 
    the outer boundary lines
    """

    def leastSquareFit(points):
        listx, listy = list(zip(*points))

        sum_xy =  0
        sum_X = 0
        n = len(listx)
        sum_x = sum(listx)
        sum_y = sum(listy)
        sumXsumY = sum_x*sum_y
        sum_x_sqr = sum_x**2
        for x,y in zip(listx,listy):
            sum_xy += x*y
            sum_X += (x**2)

        nsum_xy = n*sum_xy
        nsum_X = n*sum_X
        slope = (nsum_xy-sumXsumY)/(nsum_X-sum_x_sqr)
        intercept = ((sum_X*sum_y)-(sum_x*sum_xy))/(nsum_X-sum_x_sqr)

        return slope, intercept

    def findIntersection(line1, line2):
        ma, ca = line1[0], line1[1]
        mb, cb = line2[0], line2[1]

        if (ca, mb>0):
            ca=-ca
            mb=-mb
        else:
            ca=-(-ca)
            mb=-(-mb)
            
        x = (cb+ca)/(ma+mb)
        y = ma*x-ca
    	
        return x, y
        
    def scanPointsInLine(img, xInc, yInc):
        """
        Return the series of points in required line
        """
        img = img.T
        points = []
        width, height = img.shape[0] - 1, img.shape[1] - 1

        
        if xInc == 1 and yInc == 1:
    	    start = (0, 0)
    	    end = [width, height]
        elif xInc == 1 and yInc == -1:
            start = (0, height)
            end = [width, 0]
        elif xInc == -1 and yInc == 1:
            start = (width, 0)
            end = [0, height]
        elif xInc == -1 and yInc == -1:
            start = (width, height)
            end = [0, 0]

        for i in range(start[0], end[0], xInc):
            for j in range(start[1], end[1], yInc):
                if img[i][j] == 255:
                    points += [(i, j)]
                    end[1] = j
                    break
        
        return points
		    
#    print("I'm awesome")
    # Now, we determine the lines and intersection in the field
    pointsInLine1 = scanPointsInLine(img, 1, 1)
    pointsInLine2 = scanPointsInLine(img, 1, -1)
    pointsInLine3 = scanPointsInLine(img, -1, -1)
    pointsInLine4 = scanPointsInLine(img, -1, 1)
	
    line1 = leastSquareFit(pointsInLine1)
    line2 = leastSquareFit(pointsInLine2)
    line3 = leastSquareFit(pointsInLine3)
    line4 = leastSquareFit(pointsInLine4)


    corners = findIntersection(line1, line2), findIntersection(line2, line3), findIntersection(line3, line4), findIntersection(line4, line1)

	
    return corners
    

def getTransformationMatrix(input_points, output_img_res = (2000, 1300)):
    """
    Returns the perspective transformation matrix that transforms the input
    points (i.e. set of points in the image) into the corresponding output 
    points (i.e. Four corners of the rectangle for our case)
    """

    mat = cv2.getPerspectiveTransform(input_points, output_points)
    
    return mat

def cost(diff):
    """
    Returns the cost for given difference
    """
    return diff.mean()


def run(images, res=(2000, 1300)):
    """
    Find the transformation matrices for each camera such that the final images 
    are perfectly aligned

    Arguements :
    images = list of multiple images from different views (4-views for our case)
    res = Required resolution of the final output (projected) image


    Here, first 'image' of the input images, is taken as reference
    """

    # Number of views
    N = len(images)

    # Output points to which the input points are to be mapped
    output_points = [(0, res[1]), (res[0], res[1]), (res[0], 0), (0, 0)]

    # Images after line extraction in binary form
    bin_images = []
    # List of Corner points for each input images (view)
    input_points = []
    for img in images:
        print("Finding corners for image")
        bin_images.append(extractFieldLines(img))
        input_points.append(extractCorners(bin_images[-1]))


    print(input_points[0])
    print(input_points[1])
    print(input_points[2])
    print(input_points[3])

    
    # List of Transformation matrix for each input images (view)
    proj_mat = []
    # List of output corner points for each image (view)
    op = [output_points for i in range(4)]

    # Find the projection matrix for each image (i.e. each camera view)
    for i in range(N):
#        input_points.append(extractCorners(img))
        proj_mat.append(cv2.getPerspectiveTransform(np.float32(input_points[i]), np.float32(op[i])))

    # Take the first view as reference view
    # prefixed by 'trans' if the image is "transformed" image
    transRef = cv2.warpPerspective(bin_images[0], proj_mat[0], res)


    # Find the orientation of every other image, align them using cost function
    # and find the Projection (transformation) matrix
    for i in range(1, N):

        print("For view ", i + 1)

        # Find the values of cost function for different rotations of the output image
        cost_value = []

        for j in range(4):      # since there are only four rotations for rectangle
            transImg = cv2.warpPerspective(bin_images[i], proj_mat[i], res)

            difference = transImg - transRef

            cost_value.append(cost(difference))

            op[i].append(op[i].pop(0))
            proj_mat[i] = cv2.getPerspectiveTransform(np.float32(input_points[i]), np.float32(op[i]))
        
        # Now, find the rotation at which the cost is minimum and then find the projection matrix
        print(cost_value)
        for j in range(np.argmin(cost_value)):
            op[i].append(op[i].pop(0))

        proj_mat[i] = cv2.getPerspectiveTransform(np.float32(input_points[i]), np.float32(op[i]))

#    for i in range(4):
#        cv2.imshow("Image", cv2.warpPerspective(bin_images[i], proj_mat[i], res))
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()


    return bin_images, proj_mat

