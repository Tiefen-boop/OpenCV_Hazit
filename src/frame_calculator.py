import itertools
import os
import sys
import getopt
import threading
import numpy as np
import cv2
import Gradient_Stage
import Hough_Space_Stage
import Lines_Stage
import helpFunctions
import line_unique_functions
from tqdm import tqdm


def thread_main(image, mask, gradient_computation_method, hough_space_computation_method, color,
                get_threshold_method=Hough_Space_Stage.get_threshold):
    # moving to correct working directory (and cleaning it)
    grad_dir = Gradient_Stage.METHOD_TO_NAME[gradient_computation_method]

    space_dir = Hough_Space_Stage.METHOD_TO_NAME[hough_space_computation_method]
    threshold_dir = Hough_Space_Stage.GRADIANT_THRESHOLD_TO_NAME[get_threshold_method]
    wd = helpFunctions.build_working_dir(grad_dir, space_dir, threshold_dir)

    # performing computations
    gradient = Gradient_Stage.main(gradient_computation_method, image, mask)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hough_space = Hough_Space_Stage.main(hough_space_computation_method, gradient, get_threshold_method, grad_dir)
    # saving computed data for future runs
    np.savetxt(wd + '/gradient.txt', gradient, fmt='%.0f')
    np.savetxt(wd + '/hough_space.txt', hough_space, fmt='%.0f')
    cv2.imwrite(wd + '/gradient.png', gradient)
    cv2.imwrite(wd + '/hough_space.png', hough_space)
    cv2.imwrite(wd + '/normalized_hough_space.png', hough_space * 255 / hough_space.max())

    # for each uniqueness method
    for i in tqdm(range(len(line_unique_functions.ALL_METHODS)), desc="calculating different uniqueness methods for: "
                                                                      "gradiant method: " + Gradient_Stage.METHOD_TO_NAME[gradient_computation_method] + ".space method: " + Hough_Space_Stage.METHOD_TO_NAME[hough_space_computation_method]):
        uniqueness_method = line_unique_functions.ALL_METHODS[i]
        images = [image]
        titles = ["Original"]
        for method in Lines_Stage.ALL_METHODS:
            images.append(
                Lines_Stage.main(image, gradient, hough_space, method, color, method_line_uniqueness=uniqueness_method))
            titles.append(Lines_Stage.METHOD_TO_NAME[method])
        helpFunctions.plot_images(images, titles, show=False,

                                  dir_to_save=wd + "/" + line_unique_functions.METHOD_TO_NAME[uniqueness_method])


def main(argv):
    image_addr = None
    image = None
    mask_addr = None
    mask = None
    color = (255, 0, 0)  # default red color for drawn lines
    try:
        opts, args = getopt.getopt(argv, "hi:m:", ["image=", "mask="])
    except getopt.GetoptError:
        print('test.py -i <input_image> [-m <input_mask>] [-c red|green|blue]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <input_image> [-m <input_mask>] [-c red|green|blue]')
            sys.exit()
        elif opt == "-i" or opt == "--image":
            image = cv2.imread(arg)
            image_addr = arg.split('/')[-1].split('\\')[-1]
        elif opt == "-m" or opt == "--mask":
            mask = cv2.imread(arg)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_addr = arg.split('/')[-1].split('\\')[-1]
        elif opt == "-c" or opt == "--color":
            if arg == "red" or arg == "r":
                color = (255, 0, 0)
            elif arg == "green" or arg == "g":
                color = (0, 255, 0)
            elif arg == "blue" or arg == "b":
                color = (0, 0, 255)
    if image is None:
        print('No Image: test.py -i <input_image> [-m <input_mask>] [-c red|green|blue]')
        sys.exit(2)

    image_dir = "Computation_For_" + image_addr
    if mask is not None:
        image_dir = image_dir + "_Given_mask_" + mask_addr
    wd = helpFunctions.build_working_dir(image_dir, exist_ok=True)
    os.chdir(wd)

    gradient_computation_methods = [Gradient_Stage.compute_gradient, Gradient_Stage.compute_absolute_gradient]

    # threshold_computation_methods = [Hough_Space_Stage.threshold_by_percentile,
    #                                 Hough_Space_Stage.threshold_by_uniques_median_div2]
    threshold_computation_methods = [Hough_Space_Stage.threshold_by_percentile]

    space_computation_methods = [Hough_Space_Stage.compute_hough_space_1_optimized,
                                 Hough_Space_Stage.compute_hough_space_2]

    threads = []
    for grad_method, space_method, threshold_computation_method in itertools.product(gradient_computation_methods,
                                                                                     space_computation_methods,
                                                                                     threshold_computation_methods):
        thread = threading.Thread(target=thread_main,
                                  args=(image, mask, grad_method, space_method, color, threshold_computation_method))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main(sys.argv[1:])
