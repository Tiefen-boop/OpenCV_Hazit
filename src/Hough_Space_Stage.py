import math
import numpy as np
import itertools
from tqdm import tqdm

import Gradient_Stage


def get_threshold(gradient=None):
    return 10


# def get_median_threshold(gradient):
#     gradient = Gradient_Stage.filter_gradient(gradient, 0)
#     return threshold_by_percentile(gradient)


def thres_1(gradient):
    gradient = np.unique(gradient)
    gradient = [x for x in gradient if x > 0]
    print(np.median(gradient))
    return np.median(gradient)


def thres_2(gradient):
    gradient = [x for x in gradient if x > 0]
    print(np.max(gradient) / 4)
    return np.max(gradient) / 4


def threshold_by_uniques_median_div2(gradient):
    gradient = np.unique(gradient)
    gradient = [x for x in gradient if x >= 0]
    print(np.median(gradient) / 2)
    return np.median(gradient) / 2


# todo when not using a mask, a percentage of 0.05 is reccomended
def threshold_by_percentile(gradient, top_percentage=0.1):
    gradient = gradient.flatten()
    gradient = [x for x in gradient if x > 0]
    index = int(len(gradient) * top_percentage)
    threshold = np.partition(gradient, -index)[-index]
    print(threshold)
    return threshold


def mat_to_vector_of_relevant_points(gradient, threshold=0):
    points = np.array([[x, y] for y, x in itertools.product(range(len(gradient)), range(len(gradient[0]))) if
                       gradient[y][x] > threshold], dtype=int)
    return points


def compute_hough_space_1_optimized(gradient, method=get_threshold, gradient_method_name=Gradient_Stage.METHOD_TO_NAME[Gradient_Stage.ALL_METHODS[0]]):
    points = mat_to_vector_of_relevant_points(gradient, method(gradient))
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((r_max, theta_max))
    for p1 in tqdm(range(len(points)),desc="Computing Hough Space O(n^2) for gradient method "+str(gradient_method_name)+"..."):
        [x1, y1] = points[p1]
        for p2 in range(p1 + 1, len(points)):
            [x2, y2] = points[p2]
            if x1 == x2:
                theta = 0
                r = x1
            else:
                m = (y2 - y1) / (x2 - x1)
                coefs = [m, y1 - m * x1]  # returns [c1, c0] for y=c1*x + c0
                # coefs = np.polyfit([x1, x2], [y1, y2], 1)
                r = int(np.abs(coefs[1]) / np.sqrt((coefs[0] * coefs[0]) + 1))
                alpha = int(np.arctan(coefs[0]) * 180 / np.pi)
                if coefs[1] < 0:
                    theta = - (90 - alpha)
                else:
                    theta = 90 + alpha
            theta = (theta + 180) % 360  # rotate theta
            hough_space[r][theta] = hough_space[r][theta] + 1
    hough_space = hough_space  # * 255 / hough_space.max()
    return hough_space


def compute_hough_space_1_optimized2(gradient, method=get_threshold):
    points = mat_to_vector_of_relevant_points(gradient, method(gradient))
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((r_max, theta_max))
    for p1, p2 in itertools.combinations(points, 2):
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            theta = 0
        else:
            m = (y2 - y1) / (x2 - x1)
            coefs = [m, y1 - m * x1]
            r = int(np.abs(coefs[1]) / np.sqrt((coefs[0] * coefs[0]) + 1))
            alpha = int(np.arctan(coefs[0]) * 180 / np.pi)
            if coefs[1] < 0:
                theta = - (90 - alpha)
            else:
                theta = 90 + alpha
        theta_rad = theta * np.pi / 180
        r = np.abs(int((x1 * np.cos(theta_rad)) + (y1 * np.sin(theta_rad))))
        theta = (theta + 180) % 360  # rotate theta
        hough_space[r][theta] = hough_space[r][theta] + 1  # (gradient[y1][x1] + gradient[y2][x2])
    hough_space = hough_space  # * 255 / hough_space.max()
    return hough_space


def compute_hough_space_2(gradient, method=get_threshold, gradient_method_name=Gradient_Stage.METHOD_TO_NAME[Gradient_Stage.ALL_METHODS[0]]):
    points = mat_to_vector_of_relevant_points(gradient, method(gradient))
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((r_max, theta_max))
    for p in tqdm(points, desc="Computing Hough Space O(n) for gradient method "+str(gradient_method_name)+"..."):
        x, y = p
        for theta in range(-89, 179):
            theta_rad = theta * np.pi / 180
            r = np.abs(int((x * np.cos(theta_rad)) + (y * np.sin(theta_rad))))
            theta = (theta + 180) % 360  # rotate theta
            hough_space[r][theta] = hough_space[r][theta] + 1  # + gradient[y][x]
    hough_space = hough_space  # * 255 / hough_space.max()
    return hough_space


def compute_hough_space_1_optimized_considering_gradient(gradient, method=get_threshold):
    points = mat_to_vector_of_relevant_points(gradient, method(gradient))
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((r_max, theta_max))
    for p1 in tqdm(range(len(points))):
        [x1, y1] = points[p1]
        for p2 in range(p1 + 1, len(points)):
            [x2, y2] = points[p2]
            if x1 == x2:
                theta = 0
                r = x1
            else:
                m = (y2 - y1) / (x2 - x1)
                coefs = [m, y1 - m * x1]  # returns [c1, c0] for y=c1*x + c0
                # coefs = np.polyfit([x1, x2], [y1, y2], 1)
                r = int(np.abs(coefs[1]) / np.sqrt((coefs[0] * coefs[0]) + 1))
                alpha = int(np.arctan(coefs[0]) * 180 / np.pi)
                if coefs[1] < 0:
                    theta = - (90 - alpha)
                else:
                    theta = 90 + alpha
            theta = (theta + 180) % 360  # rotate theta
            hough_space[r][theta] = hough_space[r][theta] + (gradient[y1][x1] + gradient[y2][x2]) / 255
    hough_space = hough_space  # * 255 / hough_space.max()
    return hough_space


# using computation_method to produce a hough space for given gradient
def main(computation_method, gradient, method=get_threshold, gradient_method_name=Gradient_Stage.METHOD_TO_NAME[Gradient_Stage.ALL_METHODS[0]]):
    return computation_method(gradient, method,gradient_method_name)


# constants for this stage
ALL_METHODS = [compute_hough_space_1_optimized, compute_hough_space_1_optimized2, compute_hough_space_2,compute_hough_space_1_optimized_considering_gradient]

METHOD_TO_NAME = {
    compute_hough_space_1_optimized: "O(n^2)",
    compute_hough_space_1_optimized_considering_gradient: "O(n^2) considering gradient values",
    compute_hough_space_1_optimized2: "O(n^2)_alt",
    compute_hough_space_2: "O(n)"
}
ALL_GRADIANT_THRESHOLD_METHODS = [get_threshold, threshold_by_uniques_median_div2, threshold_by_percentile]
GRADIANT_THRESHOLD_TO_NAME = {
    get_threshold: "threshold_" + str(get_threshold()),
    # get_median_threshold: "median_threshold"
    threshold_by_uniques_median_div2: "median_threshold_div2",
    threshold_by_percentile: "percentile_threshold"
}
