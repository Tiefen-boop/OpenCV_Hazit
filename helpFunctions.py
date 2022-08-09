import math
from collections import defaultdict

import cv2
import numpy as np
import itertools
from tqdm import tqdm

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


def mat_to_vector_of_relevant_points_1(gradient, param=0):
    points = []
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((r_max, theta_max))
    for x1 in range(x_max):
        for y1 in range(y_max):
            if gradient[y1][x1] != 0:
                points.append([y1, x1])
    return points


def mat_to_vector_of_relevant_points_2(gradient, threshold=0):
    points = np.array([[x, y] for y, x in itertools.product(range(len(gradient)), range(len(gradient[0]))) if
                       gradient[y][x] > threshold], dtype=int)
    return points  # np.ndarray.flatten(points)


# @jit
# def compute_hough_space_1(gradient):
#     y_max = len(gradient)
#     x_max = len(gradient[0])
#     r_max = int(math.hypot(x_max, y_max))
#     theta_max = 360
#     hough_space = np.zeros((r_max, theta_max))
#     for x1 in range(x_max):
#         for y1 in range(y_max):
#             if gradient[y1][x1] != 0:
#                 for x2 in range(x1, x_max):
#                     for y2 in range(0, y_max):
#                         if gradient[y2][x2] != 0 and (x1 != x2 or y1 != y2):
#                             if x1 == x2:
#                                 theta = 0
#                                 r = x1
#                             else:
#                                 coefs = np.polyfit([x1, x2], [y1, y2], 1)
#                                 r = int(np.abs(coefs[1]) / np.sqrt((coefs[0] * coefs[0]) + 1))
#                                 alpha = int(np.arctan(coefs[0]) * 180 / np.pi)
#                                 if coefs[1] < 0:
#                                     theta = - (90 - alpha)
#                                 else:
#                                     theta = 90 + alpha
#                             theta = (theta + 180) % 360  # rotate theta
#                             hough_space[r][theta] = hough_space[r][theta] + 1
#     hough_space = hough_space * 255 / hough_space.max()
#     return hough_space
#
#
def compute_hough_space_1_optimized(gradient):
    points = mat_to_vector_of_relevant_points_1(gradient, 0)
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 360
    hough_space = np.zeros((r_max, theta_max))
    for p1 in tqdm (range(len(points))):
        [y1, x1] = points[p1]
        for p2 in range(p1 + 1, len(points)):
            [y2, x2] = points[p2]
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




def compute_hough_space_1_optimized2(gradient):
    points = mat_to_vector_of_relevant_points_2(gradient, 0)
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
            r = x1
        else:
            coefs = np.polyfit([x1, x2], [y1, y2], 1)
            r = int(np.abs(coefs[1]) / np.sqrt((coefs[0] * coefs[0]) + 1))
            alpha = int(np.arctan(coefs[0]) * 180 / np.pi)
            if coefs[1] < 0:
                theta = - (90 - alpha)
            else:
                theta = 90 + alpha
        theta_rad = theta * np.pi / 180
        r = int((x1 * np.cos(theta_rad)) + (y1 * np.sin(theta_rad)))
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



# def compute_hough_space_2_optimized(gradient):
#     points = matToVectorOfRelevantPoints_2(gradient, 0)
#     y_max = len(gradient)
#     x_max = len(gradient[0])
#     r_max = int(math.hypot(x_max, y_max))
#     theta_max = 180
#     hough_space = np.zeros((r_max, theta_max))
#     for p in points:
#         x, y = p
#         for theta in range(0, 179):
#             theta_rad = theta * np.pi / 180
#             r = int((x * np.cos(theta_rad)) + (y * np.sin(theta_rad)))
#             hough_space[r][theta] = hough_space[r][theta] + 1
#     hough_space = hough_space * 255 / hough_space.max()
#     return hough_space


def createLineIterator(points, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability
    P1 = points[0]
    P2 = points[1]
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


def findCoordinatesOfMaxValues(matrix, amountOfValues):
    flatted = matrix.flatten()
    indexes = np.argpartition(flatted, -1 * amountOfValues)[-1 * amountOfValues:]
    # topValues=flatted[indexes]
    """
        return in [x,y] format
    """
    matrixIndexes = [np.array([int(index % len(matrix[0])), int(index / len(matrix[0]))]) for index in indexes]
    return matrixIndexes


def findLineTwoVer(matrix, coordinate):
    r = coordinate[0]
    theta =( coordinate[1] - 180)% 360
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    y_max = len(matrix)
    x_max = len(matrix[0])
    alpha = 0
    if theta == 0:  # line is of the form X=const
        return [np.array([x, 0]), np.array([x, y_max])]  # [p1,p2]
    if theta == 90:  # line is of the form y=const
        return [np.array([0, y]), np.array([x_max, y])]  # [p1,p2]
    if theta > 0:
        alpha = 90 - theta
    if theta < 0:
        alpha = 90 + theta  # also equal 90-abs(theta)
    m = np.arctan(alpha)
    b = y - m * x
    # y=mx+b
    if theta > 0 and theta < 90:
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


def findMaxValuedLines(matrix, amountOfLines=20):
    lines = [createLineIterator(findLineTwoVer(matrix, coordinate), matrix)
             for coordinate in findCoordinatesOfMaxValues(matrix, amountOfLines)]
    return lines

def compute_hough_space_2_optimized(gradient):
    points = mat_to_vector_of_relevant_points_2(gradient, 0)
    y_max = len(gradient)
    x_max = len(gradient[0])
    r_max = int(math.hypot(x_max, y_max))
    theta_max = 180
    hough_space = np.zeros((2 * r_max, theta_max))
    for p in points:
        x, y = p
        for theta in range(0, 179):
            theta_rad = theta * np.pi / 180
            r = int((x * np.cos(theta_rad)) + (y * np.sin(theta_rad))) + r_max
            hough_space[r][theta] = hough_space[r][theta] + 1
    hough_space = hough_space * 255 / hough_space.max()
    return hough_space

