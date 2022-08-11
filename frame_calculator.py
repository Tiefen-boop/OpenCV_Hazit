import getopt
import sys
import threading

import numpy as np

import findingLinesByhoughSpace
from findingLinesByhoughSpace import continue_hough_space
from helpFunctions import *


def main(argv):
    image_found = False
    image = None
    mask_found = False
    mask = None
    try:
        opts, args = getopt.getopt(argv, "hi:m:", ["image=", "mask="])
    except getopt.GetoptError:
        print('test.py -i <input_image> [-m <input_mask>]')
        sys.exit(2)
    for opt, arg in opts:
        match opt:
            case '-h':
                print('test.py -i <input_image> [-m <input_mask>]')
                sys.exit()
            case "-i" | "--image":
                image_found = True
                image = cv2.imread(arg)
            case "-m" | "--mask":
                mask_found = True
                mask = cv2.imread(arg)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if not image_found:
        print('no image: test.py -i <input_image> [-m <input_mask>]')
        sys.exit(2)
    # image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # laplacian
    ddepth = cv2.CV_64F
    kernel_size = 3
    window_name = "laplace demo"
    laplaced = cv2.Laplacian(gray, ddepth, ksize=kernel_size)  # a matrix
    laplacians = [laplaced, np.abs(laplaced)]
    # continue_hough_space([laplaced])

    # mask
    masked = laplacians
    if mask_found:
        masked = [apply_mask(laplaced, mask) for laplaced in laplacians]

    # hough space computation
    computation_methods = [compute_hough_space_1_optimized2, compute_hough_space_2]
    threads = np.array([None] * (len(masked) * len(computation_methods)), dtype=object)
    hough_spaces = np.array([None] * (len(masked) * len(computation_methods)), dtype=object)
    for i in range(0, len(masked), len(computation_methods)):
        for j in range(len(computation_methods)):
            threads[i + j] = threading.Thread(target=computation_methods[j],
                                              args=(masked[i], hough_spaces, i + j))
            threads[i + j].start()
    for thread in np.ndarray.flatten(threads):
        thread.join()
    # np.savetxt('laplaced.txt', laplaced, fmt='%.0f')
    # np.savetxt('hough_space.txt', hough_space, fmt='%.0f')
    # images = [image, gray, laplaced, masked, hough_space]
    # continue_hough_space(images)
    # findingLinesByhoughSpace.main(image, laplaced, hough_space)
    hough_spaces.shape = (len(laplacians), len(computation_methods))
    findingLinesByhoughSpace.main2(image, laplacians, hough_spaces)


if __name__ == "__main__":
    main(sys.argv[1:])
