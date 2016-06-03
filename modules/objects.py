"""
This module implements the basic methods used to find the position of the vehicle in the field
It uses the camera matrices for projection, that can be found using calibrate module
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

from . import merge, calibrate



class Camera:
    """
    'Camera' class holds all the information about a camera.
    The camera projection matrix, the camera reference view, intrinsic parameters if required
    """
    def __init__(self, projMatrix=None, refView=None):
        """
        Initialize the camera projection matrix and camera reference view
        """
        self.projMatrix = projMatrix
        self.refView = refView

    def setProjMatrix(self, projMatrix):
        """
        Sets the camera projection matrix
        """
        self.projMatrix = projMatrix

    def setRefView(self, refView):
        """
        Sets the reference view for the camera
        """
        self.refView = refView

    def getFrameDifference(self, newFrame):
        """
        returns the difference between newFrame and reference frame
        """
        x = cv2.absdiff(newFrame, self.refView)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        ret, x = cv2.threshold(x, 2, 255, cv2.THRESH_BINARY)
        return x

    pass


class Field:
    """
    This class represents the field in which reference views from each camera are stored, along with the reconsturcted top-view

    It includes all the methods required to find the position of the vehicle in the field
    """
    def __init__(self, nCamera=None, res=(1000, 550)):
        """
        Initialize the field with number of cameras to be used along with the field resolution for reconstructed view
        """
        assert nCamera != None

        self.nCamera = nCamera
        self.camera = [Camera() for i in range(nCamera)]

        self.res = res

        pass

    def initCameraParameters(self, refViews=None):
        """
        Find the camera projection matrix and set the camera reference view

        refViews = list of reference images for all cameras in order
        """
        assert refViews != None

        projMat = calibrate.run(refViews)

        for i in range(self.nCamera):
            self.camera[i].setProjMatrix(projMat[i])
            self.camera[i].setRefView(refViews[i])

        return projMat, refViews


    def reconstruct(self, views=None):
        """
        Reconstructs the field i.e. Top-View
        """
        if views == None:
            projectedViews = [cv2.warpPerspective(self.camera[i].refView, self.camera[i].projMatrix, self.res) for i in range(self.nCamera)]

            ret = self.topView = merge.useAverage(projectedViews)

        else:
            projectedViews = [cv2.warpPerspective(views[i], self.camera[i].projMatrix, self.res) for i in range(self.nCamera)]

            ret = merge.useAverage(projectedViews)

        return ret


    def findObjectPosition(self, views=None):
        """
        Use reference frames to get the background removal from input views
        and then projected using camera matrices then, intesections are found
        to get the position of object in the field
        """
        assert views != None

        views_bgless = [self.camera[i].getFrameDifference(views[i]) for i in range(self.nCamera)]

        views_projected = [cv2.warpPerspective(views_bgless[i], self.camera[i].projMatrix, self.res) for i in range(self.nCamera)]

        return merge.useAnd(views_projected)


    pass


def main():
    pass

if __name__ == '__main__':
    main()
