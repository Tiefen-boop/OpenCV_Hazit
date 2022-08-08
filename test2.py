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



matrix=np.array([[1,2,10],[1,4,3],[1,9,3],[20,2,3],[1,2,3]],dtype=int)
np.savetxt('test.txt', matrix)
with open("test.txt") as textFile:
    lines = [line.split() for line in textFile]
# image
imgAddress = "images/imagesForTesting/image.jpg"
img = cv2.imread(imgAddress)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0)

# laplacian
ddepth = cv2.CV_64F
kernel_size = 3
window_name = "laplace demo"
laplaced = cv2.Laplacian(gray, ddepth, ksize=kernel_size)  # a matrix

# filter
# filtered = helpFunctions.filter_gradient(laplaced, 0)
filtered = laplaced

# mask
maskAddress = "imageParal.jpg"
mask = cv2.imread(maskAddress)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#masked = helpFunctions.apply_mask(filtered, mask)
masked = filtered


from alive_progress import alive_bar


hough_space = helpFunctions.compute_hough_space_1_optimized2(masked)
np.savetxt('hough_space.txt', hough_space)
# with open("test.txt") as textFile:
#     lines = [line.split() for line in textFile]
lines=helpFunctions.findMaxValuedLines(hough_space,2)
print(lines)

titles = ['original image', 'gray', 'laplaced', 'filtered', 'masked', 'hough space']
images = [img, gray, laplaced, filtered, masked, hough_space]
for i in range(len(images)):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
