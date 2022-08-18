import itertools
import os
import shutil
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


def thread_main(image, mask, gradient_computation_method, hough_space_computation_method,get_threshold_method=Hough_Space_Stage.get_threshold):
    # moving to correct working directory (and cleaning it)
    grad_dir = Gradient_Stage.METHOD_TO_NAME[gradient_computation_method]
    os.makedirs(grad_dir, exist_ok=True)
    space_dir = grad_dir + "/" + Hough_Space_Stage.GRADIANT_THRESHOLD_TO_NAME[get_threshold_method] +"/" + Hough_Space_Stage.METHOD_TO_NAME[hough_space_computation_method]
    if os.path.exists(space_dir):  # optional
        shutil.rmtree(space_dir)   # optional
    os.makedirs(space_dir)

    # performing computations
    gradient = Gradient_Stage.main(gradient_computation_method, image, mask)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hough_space = Hough_Space_Stage.main(hough_space_computation_method, gradient)
    # saving computed data for future runs
    np.savetxt(space_dir + '/gradient.txt', gradient, fmt='%.0f')
    np.savetxt(space_dir + '/hough_space.txt', hough_space, fmt='%.0f')
    cv2.imwrite(space_dir + '/gradient.png', gradient)
    cv2.imwrite(space_dir + '/hough_space.png', hough_space)
    cv2.imwrite(space_dir + '/normalized_hough_space.png', hough_space * 255 / hough_space.max())


    #for each uniqueness method
    for uniqueness_method in line_unique_functions.ALL_METHODS:
        images = [image, gradient]
        titles = ["Original", "Gradient"]
        for method in Lines_Stage.ALL_METHODS:
            images.append(Lines_Stage.main(image, gradient, hough_space, method, method_line_uniqueness=uniqueness_method))
            titles.append(Lines_Stage.METHOD_TO_NAME[method])
        helpFunctions.plot_images(images, titles, show=False, dir_to_save=space_dir+"/" + line_unique_functions.METHOD_TO_NAME[uniqueness_method])




def main(argv):
    image_addr = None
    image = None
    mask_addr = None
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
                image = cv2.imread(arg)
                image_addr = arg
            case "-m" | "--mask":
                mask = cv2.imread(arg)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_addr = arg
    if image is None:
        print('no image: test.py -i <input_image> [-m <input_mask>]')
        sys.exit(2)

    image_dir = "Computation_For_" + image_addr.split('/')[-1]
    if mask is not None:
        image_dir = image_dir + "_Given_mask_" + mask_addr.split('/')[-1]
    os.makedirs(image_dir, exist_ok=True)
    os.chdir(image_dir)

    gradient_computation_methods = [Gradient_Stage.compute_gradient, Gradient_Stage.compute_absolute_gradient]
    space_computation_methods = [Hough_Space_Stage.compute_hough_space_1_optimized,
                                 Hough_Space_Stage.compute_hough_space_2]
    threshold_computation_methods = [Hough_Space_Stage.get_threshold, Hough_Space_Stage.get_median_threshold]
    # space_computation_methods = [Hough_Space_Stage.compute_hough_space_1_optimized]
    # space_computation_methods = [Hough_Space_Stage.compute_hough_space_2]

    threads = []
    for grad_method, space_method, threshold_computation_method in itertools.product(gradient_computation_methods, space_computation_methods, threshold_computation_methods):
        thread = threading.Thread(target=thread_main, args=(image, mask, grad_method, space_method, threshold_computation_method))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main(sys.argv[1:])
