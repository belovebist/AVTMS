"""
This is a test script for merge routines
"""
import merge
import cv2
from matplotlib import pyplot as plt


def main():
    """
    Takes multiple images as input, projectively transform and merge them
    """

    image_file = "../Images/field.png"

    img1 = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_file, cv2.IMREAD_COLOR)
    
    plt.subplot(2, 2, 1), plt.imshow(img1, cmap = 'gray'), plt.title("Gray")

    img1 = cv2.merge((img1, img1, img1))

    ret = merge.average((img1, img2))

    plt.subplot(2, 2, 2), plt.imshow(img2[:, :, -1::-1]), plt.title("Color")
    plt.subplot(2, 2, 3), plt.imshow(ret[:, :, -1::-1]), plt.title("Merged")

    plt.show()


    pass


if __name__ == '__main__':
    main()
