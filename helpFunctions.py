import math
from collections import defaultdict

import cv2
import numpy as np


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


def findVer(vertices):
    cleaned = [ver[0] for ver in vertices]
    fourVer = [[0, 0], [0, 0], [0, 0], [0, 0]]
    leftDown = [0, 0]
    leftUp = [0, 0]
    rightDown = [0, 0]
    rightUp = [0, 0]
    for x, y in cleaned:
        True


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


def compute_hough_space(gradient):
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((r_max, theta_max))
    for x1 in range(x_max):
        for y1 in range(y_max):
            if gradient[y1][x1] != 0:
                for x2 in range(x1, x_max):
                    for y2 in range(y1, y_max):
                        if gradient[y2][x2] != 0 and (x1 != x2 or y1 != y2):
                            if x1 == x2:
                                theta = 0
                                r = x1
                            else:
                                coefs = np.polyfit([x1, x2], [y1, y2], 1)
                                r = int(np.abs(coefs[1]) / np.sqrt((coefs[0] * coefs[0]) + 1))
                                alpha = int(np.arctan(coefs[0]) * 180 / np.pi)
                                if coefs[1] < 0:
                                    theta = - (90 - alpha)
                                else:
                                    theta = 90 + alpha
                            theta = (theta + 180) % 360  # rotate theta
                            hough_space[r][theta] = hough_space[r][theta] + 1
    hough_space = hough_space * 255 / hough_space.max()
    return hough_space


def compute_hough_space_2(gradient):
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((r_max, theta_max))
    for x in range(x_max):
        for y in range(y_max):
            if gradient[y][x] != 0:
                for theta in range(-89, 179):
                    theta_rad = theta * np.pi / 180
                    r = int((x * np.cos(theta_rad)) + (y * np.sin(theta_rad)))
                    theta = (theta + 180) % 360  # rotate theta
                    hough_space[r][theta] = hough_space[r][theta] + 1
    hough_space = hough_space * 255 / hough_space.max()
    return hough_space
