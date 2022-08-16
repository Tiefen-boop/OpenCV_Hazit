import copy

import cv2
import numpy as np

from helpFunctions import cut_to_intersection


def get_threshold(gradient):
    return 0


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


def find_max_valued_lines(hough_space, gradient, amount_of_lines=20):
    lines = [create_line_iterator(find_line_two_ver(gradient, coordinate), gradient)
             for coordinate in find_coordinates_of_max_values(hough_space, amount_of_lines)]
    # lines = limit_lines_to_relevant_edges(lines)
    return lines


# Scoring Methods:
def score_by_gradients_quality(line, gradient):
    threshold = get_threshold(gradient)
    return np.sum(list(filter(lambda x: x >= threshold, line[:, 2]))) / line.size if line.size > 0 else -1


# calculates the amount of values in the line that <=threshold
def score_by_density(line, gradient):
    threshold = get_threshold(gradient)
    return np.count_nonzero(line[:, 2] <= threshold) * -1 / line.size if line.size > 0 else -1000000000


def score_by_frequency(line, gradient):
    threshold = get_threshold(gradient)
    score = 0
    points = 0
    for val in line[:, 2]:
        if val > threshold:
            points = 0
        else:
            points -= 1
            score += points
    return score / line.size if line.size > 0 else -1000000000


def score_by_frequency2(line, gradient):
    threshold = get_threshold(gradient)
    score = 0
    current_sequence_size = 0
    for val in line[:, 2]:
        if val > threshold:
            if current_sequence_size > 0:
                if current_sequence_size % 2 == 0:
                    sum_of_seq = (current_sequence_size / 2 * (1 + current_sequence_size / 2))  # n*(a1+an)/2. when
                    # n=current_sequence_size/2, a1=1, an=current_sequence_size/2.
                    # notice we don't divide by 2 because look at the sum of 1,2,3,3,2,1
                else:
                    sum_of_seq = current_sequence_size / 2 * (
                            1 + current_sequence_size / 2) + current_sequence_size / 2 + 1
                score -= sum_of_seq
                current_sequence_size = 0
        else:
            current_sequence_size += 1
    return score / line.size if line.size > 0 else -1000000000


def draw_all_lines(img, lines):
    for line in lines:
        # line=np.array(line)
        if line.size == 0:
            continue
        p1 = line[0]
        p2 = line[-1]
        cv2.line(img, (round(p1[0]), round(p1[1])), (round(p2[0]), round(p2[1])), (0, 0, 255), 2)


def limit_line_to_relevant_edges(line, threshold=1):
    start_ind = len(line)
    for pointInd in range(len(line)):
        if line[pointInd][2] >= threshold:
            start_ind = pointInd
            break
    limited_line = line[start_ind:]
    for pointInd in range(len(line) - 1, 0, -1):
        if line[pointInd][2] >= threshold:
            limited_line = limited_line[0:pointInd]
            break
    return limited_line


def cart_to_polar(line):
    x1, y1 = line[0][:2]
    x2, y2 = line[-1][:2]
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
    return [r, theta]


def is_line_unique(line, lines, max_distance=6):
    r, theta = cart_to_polar(line)
    for existing_line in lines:
        r2, theta2 = cart_to_polar(existing_line)
        if round(theta, 3) == round(theta2, 3) and abs(r - r2) < max_distance:
            return False
    return True


def get_top_lines(lines, laplaced, method):
    lines_limited = [limit_line_to_relevant_edges(line) for line in lines]
    lines_scored = np.array([[lines[i], method(lines_limited[i], laplaced)] for i in range(len(lines_limited))],
                            dtype=object)
    indices_of_top4 = np.argpartition(lines_scored[:, 1], 0)[-6:]
    top4 = lines_scored[indices_of_top4][:, 0]
    return top4


def get_top_lines_2(lines, laplaced, method,amount_of_lines=4):
    lines_limited = [limit_line_to_relevant_edges(line, 50) for line in lines]
    lines_scored = np.array([[lines[i], method(lines_limited[i], laplaced)] for i in range(len(lines_limited))],
                            dtype=object)
    indices_of_descending_values = lines_scored[:, -1].argsort()[::-1]
    lines_sorted_descending = lines_scored[indices_of_descending_values][:, 0]
    top4 = []
    for line in lines_sorted_descending:
        if line.size <= 1:
            continue
        if is_line_unique(line, top4):
            top4.append(line)
            if len(top4) == amount_of_lines:
                break
    if len(top4) < amount_of_lines:
        print("didnt find "+str(amount_of_lines)+ " lines")

    return np.array(top4, dtype=object)


# finds - based on given gradient, hough_space, scoring_method - the best lines
# returns the intersections of the lines + a copy of the image with the lines plotted on
def main(image, gradient, hough_space, scoring_method):
    lines = find_max_valued_lines(hough_space, gradient)
    top_lines = get_top_lines_2(lines, gradient, scoring_method, amount_of_lines=4)
    top_lines = cut_to_intersection(top_lines)

    drawn_image = copy.deepcopy(image)
    draw_all_lines(drawn_image, top_lines)
    return drawn_image


# constants for this stage
ALL_METHODS = [score_by_gradients_quality, score_by_density, score_by_frequency, score_by_frequency2]

METHOD_TO_NAME = {
    score_by_gradients_quality: "By Quality",
    score_by_density: "By Density",
    score_by_frequency: "By Frequency",
    score_by_frequency2: "By Frequency (alt)"
}
