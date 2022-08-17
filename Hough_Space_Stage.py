import math
import numpy as np
import itertools
from tqdm import tqdm


def get_threshold(gradient=None):
    return 20


def mat_to_vector_of_relevant_points(gradient, threshold=0):
    points = np.array([[x, y] for y, x in itertools.product(range(len(gradient)), range(len(gradient[0]))) if
                       gradient[y][x] > threshold], dtype=int)
    return points


def compute_hough_space_1_optimized(gradient):
    points = mat_to_vector_of_relevant_points(gradient, get_threshold(gradient))
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
            hough_space[r][theta] = hough_space[r][theta] + 1
    hough_space = hough_space  # * 255 / hough_space.max()
    return hough_space


def compute_hough_space_1_optimized2(gradient):
    points = mat_to_vector_of_relevant_points(gradient, get_threshold(gradient))
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((2 * r_max, theta_max))
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
        r = int((x1 * np.cos(theta_rad)) + (y1 * np.sin(theta_rad))) + r_max
        theta = (theta + 180) % 360  # rotate theta
        hough_space[r][theta] = hough_space[r][theta] + 1  # (gradient[y1][x1] + gradient[y2][x2])
    hough_space = hough_space  # * 255 / hough_space.max()
    return hough_space


def compute_hough_space_2(gradient):
    points = mat_to_vector_of_relevant_points(gradient, get_threshold(gradient))
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 180
    hough_space = np.zeros((2 * r_max, theta_max))
    for p in tqdm(points):
        x, y = p
        for theta in range(0, 179):
            theta_rad = theta * np.pi / 180
            r = int((x * np.cos(theta_rad)) + (y * np.sin(theta_rad))) + r_max
            hough_space[r][theta] = hough_space[r][theta] + 1  # + gradient[y][x]
    hough_space = hough_space  # * 255 / hough_space.max()
    return hough_space


# using computation_method to produce a hough space for given gradient
def main(computation_method, gradient):
    return computation_method(gradient)


# constants for this stage
ALL_METHODS = [compute_hough_space_1_optimized, compute_hough_space_1_optimized2, compute_hough_space_2]

METHOD_TO_NAME = {
    compute_hough_space_1_optimized: "O(n^2)",
    compute_hough_space_1_optimized2: "O(n^2)_alt",
    compute_hough_space_2: "O(n)"
}
