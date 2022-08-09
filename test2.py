import os

import cv2
import numpy as np
import helpFunctions

# Reading the required image in
# which operations are to be done.
# Make sure that the image is in the same
# directory in which this python program is

from collections import defaultdict

from matplotlib import pyplot as plt

from findingLinesByhoughSpace import continueHoughSpace

matrix=np.array([[1,2,10],[1,4,3],[1,9,3],[20,2,3],[1,2,3]],dtype=int)
np.savetxt('test.txt', matrix, fmt ='%.0f')
with open("test.txt") as textFile:
    lines = [line.split() for line in textFile]
# image
imgAddress = "images/imagesForTesting/imageHalfSize.jpg"
img = cv2.imread(imgAddress)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0)

# laplacian
ddepth = cv2.CV_64F
kernel_size = 3
window_name = "laplace demo"
laplaced = cv2.Laplacian(gray, ddepth, ksize=kernel_size)  # a matrix
np.savetxt('laplaced.txt', laplaced,fmt ='%.0f')
# filter
#filtered = helpFunctions.filter_gradient(laplaced, 220)
filtered = laplaced

# mask
maskAddress = "imageParal.jpg"
mask = cv2.imread(maskAddress)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#masked = helpFunctions.apply_mask(filtered, mask)
masked = filtered


# from alive_progress import alive_bar
#
# with alive_bar(1000) as bar:
#     for i in helpFunctions.compute_hough_space_1_optimized2(masked):
#         hough_space= bar()
hough_space = helpFunctions.compute_hough_space_1_optimized(masked)
np.savetxt('hough_space.txt', hough_space,fmt ='%.0f')
images = [img, gray, laplaced, filtered, masked, hough_space]
continueHoughSpace(images,hough_space)
