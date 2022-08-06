# Python program to illustrate HoughLine
# method for line detection
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

imgAddress="images/imagesForTesting/988Cropped.jpg"
img=cv2.imread(imgAddress)

height, width = img.shape[:2]
# Convert the img to grayscale
#img=cv2.GaussianBlur(img, (1, 1), 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# titles = ['Original Image', 'grey']
# images = [img, gray]
# for i in range(2):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

# Apply edge detection method on the image
#edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#laplacian
ddepth = cv2.cv_8uc1
kernel_size = 3
window_name = "laplace demo"
edges=cv2.laplacian(gray, ddepth, ksize=kernel_size)#a matrix


print(edges)

#filtering
for y in range (len(edges)):
    for x in range (len(edges[y])):
        if(edges[y][x]<220):
            edges[y][x] = 0




# [convert]
# converting back to uint8
abs_dst = cv2.convertScaleAbs(edges)
# [convert]
# [display]
cv2.imwrite('laplac.jpg', abs_dst)
# cv2.imshow(window_name, abs_dst)
# cv2.waitKey(0)
#print(edges)

titles = ['original image', 'grey']
images = [img, edges]
for i in range(2):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
# This returns an array of r and theta values

#lines = cv2.HoughLines(edges, 1, np.pi / 180, int( min(height,width)*0.588))
lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)



# The below for loop runs till r and theta values
# are in the range of the 2d array
print(lines)
lines = helpFunctions.lines_to_map(lines)
print("print final lines \n ")
print(lines)

for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    # Stores the value of cos(theta) in a
    a = np.cos(theta)

    # Stores the value of sin(theta) in b
    b = np.sin(theta)

    # x0 stores the value rcos(theta)
    x0 = a * r

    # y0 stores the value rsin(theta)
    y0 = b * r

    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000 * (-b))

    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000 * (a))

    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000 * (-b))

    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000 * (a))

    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# All the changes made in the input image are finally
# written on a new image houghlines.jpg


#finding the vertices
# segmented = helpFunctions.segment_by_angle_kmeans(lines)
# intersections = helpFunctions.segmented_intersections(segmented)

# for inter in intersections:
#     cv2.circle(img, (inter[0][0], inter[0][1]), radius=3, color=(0, 255, 0), thickness=-1)
#     # cv2.putText(img, str(inter[0][0])+" "+str( inter[0][1]), (inter[0][0], inter[0][1]),
#     #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
# print(intersections)




cv2.imwrite("images/linesDetected/"+imgAddress.split("/")[-1], img)


