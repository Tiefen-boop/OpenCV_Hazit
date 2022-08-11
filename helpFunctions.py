import math
import sys
from collections import defaultdict

import cv2
import numpy as np
import itertools

from matplotlib import pyplot as plt
from tqdm import tqdm


def get_first_threshold(gradient):
    return 50


def get_second_threshold(gradient):
    return 0


def lines_to_map(lines):
    cleaned = [[line[0][0], round(line[0][1], 3)] for line in lines]
    thetas = [theta for r, theta in cleaned]
    map = {}
    for theta in set(thetas):
        map[theta] = [r for r, theta2 in cleaned if theta2 == theta]
    return lines_map_to_final_lines(map)


def lines_map_to_final_lines(map, max_distance=6):
    final_lines = []
    for radian in map:
        values_for_rad = []  # this will be a 2d array
        for radius in map[radian]:  # iterates over all the values of this radius(an array of floats)
            found_value_array_for_radius = False
            for lineOfKey in values_for_rad:  # previous values found. line of key is also an array
                if abs(radius - np.average(lineOfKey)) < max_distance:
                    lineOfKey.append(radius)
                    found_value_array_for_radius = True
                    break
            if not found_value_array_for_radius:
                values_for_rad.append([radius])

        for val in values_for_rad:
            final_lines.append([[np.average(val), radian]])
    return final_lines


# def uniqueLines(lines):
#     seenLines=[]
#     addLine=True
#     for line in lines:
#         for seenline in seenLines:
#             if(round(seenline[0][1],3)==round(line[0][1],3)):
#                 addLine=False
#                 break
#         if(addLine):
#             seenLines.append(line)
#         addLine=True;
#     return seenLines

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def filter_gradient(edges, threshold=220):
    y_max = len(edges)
    x_max = len(edges[0])
    filtered = np.zeros((y_max, x_max))
    for y in range(len(edges)):
        for x in range(len(edges[y])):
            if edges[y][x] >= threshold:
                filtered[y][x] = edges[y][x]
    return filtered


def apply_mask(edges, mask):
    y_max = len(edges)
    x_max = len(edges[0])
    masked = np.zeros((y_max, x_max))
    for x in range(x_max):
        for y in range(y_max):
            if mask[y][x] != 0:
                masked[y][x] = edges[y][x]
    return masked


def mat_to_vector_of_relevant_points_1(gradient, threshold=0):
    points = []
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((r_max, theta_max))
    for x1 in range(x_max):
        for y1 in range(y_max):
            if gradient[y1][x1] > threshold:
                points.append([x1, y1])
    return points


def mat_to_vector_of_relevant_points_2(gradient, threshold=0):
    points = np.array([[x, y] for y, x in itertools.product(range(len(gradient)), range(len(gradient[0]))) if
                       gradient[y][x] > threshold], dtype=int)
    return points  # np.ndarray.flatten(points)


def compute_hough_space_1_optimized(gradient, save_to=None, at_index=None):
    points = mat_to_vector_of_relevant_points_2(gradient, get_first_threshold(gradient))
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
                coefs = [m, y1 - m * x1]
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
    if save_to is not None:
        save_to[at_index] = hough_space
    return hough_space


def compute_hough_space_1_optimized2(gradient, save_to=None, at_index=None):
    points = mat_to_vector_of_relevant_points_2(gradient, get_first_threshold(gradient))
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
        hough_space[r][theta] = hough_space[r][theta] + 1 # (gradient[y1][x1] + gradient[y2][x2])
    hough_space = hough_space  # * 255 / hough_space.max()
    if save_to is not None:
        save_to[at_index] = hough_space
    return hough_space


def compute_hough_space_2(gradient, save_to=None, at_index=None):
    points = mat_to_vector_of_relevant_points_2(gradient, get_first_threshold(gradient))
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
    if save_to is not None:
        save_to[at_index] = hough_space
    return hough_space


def create_line_iterator(points, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -p1: a numpy array that consists of the coordinate of the first point (x,y)
        -p2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability
    p1 = points[0]
    p2 = points[1]
    image_h = img.shape[0]
    image_w = img.shape[1]
    p1_x = p1[0]
    p1_y = p1[1]
    p2_x = p2[0]
    p2_y = p2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    d_x = p2_x - p1_x
    d_y = p2_y - p1_y
    d_x_a = np.abs(d_x)
    d_y_a = np.abs(d_y)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(d_y_a, d_x_a), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    neg_y = p1_y > p2_y
    neg_x = p1_x > p2_x
    if p1_x == p2_x:  # vertical line segment
        itbuffer[:, 0] = p1_x
        if neg_y:
            itbuffer[:, 1] = np.arange(p1_y - 1, p1_y - d_y_a - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(p1_y + 1, p1_y + d_y_a + 1)
    elif p1_y == p2_y:  # horizontal line segment
        itbuffer[:, 1] = p1_y
        if neg_x:
            itbuffer[:, 0] = np.arange(p1_x - 1, p1_x - d_x_a - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(p1_x + 1, p1_x + d_x_a + 1)
    else:  # diagonal line segment
        steep_slope = d_y_a > d_x_a
        if steep_slope:
            slope = d_x.astype(np.float32) / d_y.astype(np.float32)
            if neg_y:
                itbuffer[:, 1] = np.arange(p1_y - 1, p1_y - d_y_a - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(p1_y + 1, p1_y + d_y_a + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - p1_y)).astype(np.int) + p1_x
        else:
            slope = d_y.astype(np.float32) / d_x.astype(np.float32)
            if neg_x:
                itbuffer[:, 0] = np.arange(p1_x - 1, p1_x - d_x_a - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(p1_x + 1, p1_x + d_x_a + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - p1_x)).astype(np.int) + p1_y

    # Remove points outside of image
    col_x = itbuffer[:, 0]
    col_y = itbuffer[:, 1]
    itbuffer = itbuffer[(col_x >= 0) & (col_y >= 0) & (col_x < image_w) & (col_y < image_h)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


def find_coordinates_of_max_values(matrix, amount_of_values):
    flatted = matrix.flatten()
    indexes = np.argpartition(flatted, -1 * amount_of_values)[-1 * amount_of_values:]
    # topValues=flatted[indexes]
    """
        return in [x,y] format
    """
    matrix_indexes = [np.array([int(index % len(matrix[0])), int(index / len(matrix[0]))]) for index in indexes]
    return matrix_indexes


def find_line_two_ver(matrix, coordinate):
    r = coordinate[1]
    theta = (coordinate[0] - 180) % 360
    theta_rad = theta * np.pi / 180
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    y_max = len(matrix)
    x_max = len(matrix[0])
    alpha = 0
    if theta == 0:  # line is of the form X=const
        return [np.array([round(x), 0]), np.array([round(x), round(y_max)])]  # [p1,p2]
    if theta == 90:  # line is of the form y=const
        return [np.array([0, round(y)]), np.array([round(x_max), round(y)])]  # [p1,p2]
    if theta > 0:
        alpha = 90 - theta
    if theta < 0:
        alpha = 90 + theta  # also equal 90-abs(theta)
    m = np.arctan(alpha)
    b = y - m * x
    # y=mx+b
    if 0 < theta < 90:
        x2 = -b / m
        y2 = 0
        if x2 > x_max:
            x2 = x_max
            y2 = m * x2 + b
        return [np.array([0, round(b)]), np.array([round(x2), round(y2)])]  # [p1,p2]
    if theta > 90:
        y2 = m * x_max + b
        x2 = x_max
        if y2 > y_max:
            y2 = y_max
            x2 = (y2 - b) / m
        return [np.array([0, round(b)]), np.array([round(x2), round(y2)])]  # [p1,p2]
    if theta < 0:
        y2 = m * x_max + b
        x2 = x_max
        if y2 > y_max:
            y2 = y_max
            x2 = (y2 - b) / m
        return [np.array([round(-b / m), 0]), np.array([round(x2), round(y2)])]  # [p1,p2]
    return []


def find_max_valued_lines(hough_space, laplaced, amount_of_lines=20):
    lines = [create_line_iterator(find_line_two_ver(laplaced, coordinate), laplaced)
             for coordinate in find_coordinates_of_max_values(hough_space, amount_of_lines)]
    # lines = limit_lines_to_relevant_edges(lines)
    return lines


""" the function gets an array of lines an return same lines but only between the points with a gradint>threshold"""


def limit_lines_to_relevant_edges(lines, threshold=1):
    for lineInd in range(len(lines)):
        line = lines[lineInd]
        startInd = len(line)
        for pointInd in range(len(line)):
            if line[pointInd][2] >= threshold:
                startInd = pointInd
                break
        line = line[startInd:]
        lines[lineInd] = line
        for pointInd in range(len(line) - 1, 0, -1):
            if line[pointInd][2] >= threshold:
                lines[lineInd] = line[0:pointInd]
                break
    return lines


def limit_line_to_relevant_edges(line, threshold=1):
    startInd = len(line)
    for pointInd in range(len(line)):
        if line[pointInd][2] >= threshold:
            startInd = pointInd
            break
    limited_line = line[startInd:]
    for pointInd in range(len(line) - 1, 0, -1):
        if line[pointInd][2] >= threshold:
            limited_line = limited_line[0:pointInd]
            break
    return limited_line


def get_top_lines(lines, laplaced, method):
    lines_limited = [limit_line_to_relevant_edges(line) for line in lines]
    lines_scored = np.array([[lines[i], method(lines_limited[i], laplaced)] for i in range(len(lines_limited))], dtype=object)
    indices_of_top4 = np.argpartition(lines_scored[:, 1], 0)[-6:]
    top4 = lines_scored[indices_of_top4][:, 0]
    return top4


def draw_all_lines(img, lines):
    for line in lines:
        if line.size == 0:
            continue
        p1 = line[0]
        p2 = line[-1]
        cv2.line(img, (round(p1[0]), round(p1[1])), (round(p2[0]), round(p2[1])), (0, 0, 255), 2)


def score_by_gradients_quality(line, gradient):
    threshold = get_second_threshold(gradient)
    return np.sum(list(filter(lambda x: x >= threshold, line[:, 2]))) / line.size if line.size > 0 else -1


# calculates the amount of values in the line that <=threshold
def score_by_density(line, gradient):
    threshold = get_second_threshold(gradient)
    return np.count_nonzero(line[:, 2] <= threshold) * -1 / line.size if line.size > 0 else -1000000000


def plot_images(images, titles):
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
