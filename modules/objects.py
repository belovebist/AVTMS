"""
This module implements the basic methods used to find the position of the vehicle in the field
It uses the camera matrices for projection, that can be found using calibrate module
"""
import os
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
        diff = cv2.absdiff(newFrame, self.refView)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        ret, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
        return diff

    def startFeed(self, srcFile):
        """
        Start feeding the camera with the input video frames

        srcFile = name of the source file for the video
        """
        self.src = srcFile

    def getFrameGenerator(self):
        """
        Generator function that returns frames in sequence
        """
        for i in range(1500):
            fname = self.src + ('000' + str(i))[-4:] + '.png'
            if os.path.exists(fname):
                yield cv2.imread(fname, cv2.IMREAD_COLOR)
        return

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

    def initCameraParameters(self, useFrame0=True, refViews=None):
        """
        Find the camera projection matrix and set the camera reference view

        refViews = list of reference images for all cameras in order
        """
        self.getFrames = self.getFrameGenerator()

        if useFrame0 == True:
            refViews, empty = next(self.getFrames)

        projMat = calibrate.run(refViews)

        for i in range(self.nCamera):
            self.camera[i].setProjMatrix(projMat[i])
            self.camera[i].setRefView(refViews[i])

        return projMat, refViews

    def initCameraChannels(self, channel=None):
        """
        Initialize the channels that feeds the camera with input.

        For simulation : it is just a list of files of the video sources
        """
        assert channel != None

        print("\nInitializing Camera Channels : ")

        for i in range(self.nCamera):
            print('Camera', i,  'channel initialized to', channel[i])
            self.camera[i].startFeed(channel[i])

        return True



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


    def findObjectPosition(self, views=None, retWithBG=True):
        """
        Use reference frames to get the background removal from input views
        and then projected using camera matrices then, intesections are found
        to get the position of object in the field

        If retWithBG is True, the position of object drawn inside the reference field is returned
        """
        assert views != None

        views_BGless = [self.camera[i].getFrameDifference(views[i]) for i in range(self.nCamera)]

        views_projected = [cv2.warpPerspective(views_BGless[i], self.camera[i].projMatrix, self.res) for i in range(self.nCamera)]

        pos = merge.useAnd(views_projected)

        if retWithBG == True:
            return merge.useOr([self.topView, cv2.merge([pos, pos, pos])])
        else:
            return pos


    def getFrameGenerator(self):
        """
        Returns a list of new frames from each camera
        """

        # Generators that fetch frames from camera upon request
        frame_gens = [x.getFrameGenerator() for x in self.camera]

        while True:
            # List of common frames from each camera
            frames = []
            for x in frame_gens:
                try:
                    frames.append(next(x))
                except StopIteration:
                    frames.append(None)

            # Flag that indicates whether frames from all camera are present or not
            empty = not all([False if x is None else True for x in frames])

            # If not all frames are available then it returns empty=True

            if empty == True:
                break

            yield frames, empty

        # Only reaches when no more frames are present
        yield frames, empty



    def startTracking(self):
        """
        Start a general procedure that continuously tracks the position of object inside the field
        For not it outputs an image (i.e. top_view) that has track of the object
        """

        print("\n\nStarting the Tracking Process : ")
        tv = self.topView
#        temp = np.zeros(shape=tv.shape, dtype=np.uint8)

        while(True):
            frames, empty = next(self.getFrames)
            if empty == True:
                break
            obj_pos = self.findObjectPosition(frames)
#            temp = merge.useOr([tv, cv2.merge([obj_pos, obj_pos, obj_pos])])
            tv = merge.useOr([tv, obj_pos])

#        return merge.useOr([tv, temp])
        return tv


    pass


def main():
    pass

if __name__ == '__main__':
    main()
