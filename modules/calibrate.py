"""
This module contains camera calibration routines
The final result is the calibrated camera matrices that
perspectively transforms from image plane to common projection plane
(i.e. Top view)
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np



def extractCorners(img):
    """
    Extracts the corner points of the rectangular field by solving 
    the outer boundary lines
    """

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


def run(images, input_points, res=(2000, 1300)):
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

    # List of Corner points for each input images (view)
#    input_points = []
    # List of Transformation matrix for each input images (view)
    proj_mat = []
    # List of output corner points for each image (view)
    op = [output_points for i in range(4)]

    # Find the projection matrix for each image (i.e. each camera view)
    for i in range(N):
#        input_points.append(extractCorners(img))
        proj_mat.append(cv2.getPerspectiveTransform(input_points[i], np.float32(op[i])))

    # Take the first view as reference view
    # prefixed by 'trans' if the image is "transformed" image
    transRef = cv2.warpPerspective(images[0], proj_mat[0], res)

    # Find the orientation of every other image, align them using cost function
    # and find the Projection (transformation) matrix
    for i in range(1, N):

        print("For view ", i + 1)

        # Find the values of cost function for different rotations of the output image
        cost_value = []

        for j in range(4):      # since there are only four rotations for rectangle
            transImg = cv2.warpPerspective(images[i], proj_mat[i], res)

            difference = transImg - transRef

            cost_value.append(cost(difference))

            op[i].append(op[i].pop(0))
            proj_mat[i] = cv2.getPerspectiveTransform(input_points[i], np.float32(op[i]))
        
        # Now, find the rotation at which the cost is minimum and then find the projection matrix
        for j in range(np.argmin(cost_value)):
            op[i].append(op[i].pop(0))

        proj_mat[i] = cv2.getPerspectiveTransform(input_points[i], np.float32(op[i]))

    return proj_mat

