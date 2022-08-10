import cv2
import numpy as np
import helpFunctions

# Reading the required image in
# which operations are to be done.
# Make sure that the image is in the same
# directory in which this python program is

from collections import defaultdict

from matplotlib import pyplot as plt


def continue_hough_space(images, hough_space):
    # lines=helpFunctions.findMaxValuedLines(hough_space,2)
    # print(lines)
    titles = ['original image', 'gray', 'laplaced', 'filtered', 'masked', 'hough space']
    # images = [img, gray, laplaced, filtered, masked, hough_space]
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


#    with open("hough_space.txt") as textFile:
#        hough_spaceAfterReload = [line.split() for line in textFile]
#    hough_spaceAfterReload=np.array(hough_spaceAfterReload,dtype=int)
#    images=[hough_space,hough_spaceAfterReload]
#    titles = ['hough_space', 'hough_spaceAfterReload']
#    for i in range(len(images)):
#        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
#        plt.title(titles[i])
#        plt.xticks([]), plt.yticks([])
#    plt.show()

if __name__ == '__main__':
    with open("hough_space.txt") as textFile:
        hough_space = [line.split() for line in textFile]
    hough_space = np.array(hough_space, dtype=int)
    imgAddress = "images/imagesForTesting/988Cropped.jpg"
    img = cv2.imread(imgAddress)
    with open("laplaced.txt") as textFile:
        laplaced = [line.split() for line in textFile]
    laplaced = np.array(laplaced, dtype=int)
    titles = ['original image', 'hough_space', 'laplaced']
    images = [img, hough_space, laplaced]
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    lines = helpFunctions.find_max_valued_lines(hough_space, laplaced, 10)
    print(lines)
    helpFunctions.draw_all_lines(img, lines)
