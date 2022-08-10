import getopt
import sys

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

    # filter
    # filtered = helpFunctions.filter_gradient(laplaced, 220)
    filtered = laplaced

    # mask
    masked = filtered
    if mask_found:
        masked = apply_mask(filtered, mask)

    hough_space = compute_hough_space_1_optimized(masked)
    np.savetxt('hough_space.txt', hough_space, fmt='%.0f')
    images = [image, gray, laplaced, filtered, masked, hough_space]
    continue_hough_space(images, hough_space)


if __name__ == "__main__":
    main(sys.argv[1:])
