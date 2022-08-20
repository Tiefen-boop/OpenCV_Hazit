import numpy as np

with open(
        "../Computation_For_01562.png_Given_mask_01562_mask.png/Normal_Gradient/O(n^2)/percentile_threshold/hough_space.txt") as textFile:
    hough_space = [line.split() for line in textFile]
hough_space = np.array(hough_space, dtype=int)
print('a')
