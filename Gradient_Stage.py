import numpy as np
import cv2


def convert_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def calculate_gradient(grayscale_image):
    ddepth = cv2.CV_64F
    kernel_size = 3
    return filter_gradient(cv2.Laplacian(grayscale_image, ddepth, ksize=kernel_size))


def filter_gradient(edges, threshold=0):  # todo change to list comprehension
    y_max = len(edges)
    x_max = len(edges[0])
    filtered = np.zeros((y_max, x_max))
    for y in range(len(edges)):
        for x in range(len(edges[y])):
            if edges[y][x] >= threshold:
                filtered[y][x] = edges[y][x]
    return filtered


def apply_mask(gradient, mask):
    if mask is None:
        return gradient
    y_max = len(gradient)
    x_max = len(gradient[0])
    masked = np.zeros((y_max, x_max))
    for x in range(x_max):
        for y in range(y_max):
            if mask[y][x] != 0:
                masked[y][x] = gradient[y][x]
    return masked


def absolute_gradient(gradient):
    return np.abs(gradient)


# performs all calculation in order to compute the gradient
def compute_gradient(image, mask):
    grayscale = convert_to_gray(image)
    gradient = calculate_gradient(grayscale)
    return apply_mask(gradient, mask)


# performs all calculation in order to compute the absolute value gradient
def compute_absolute_gradient(image, mask):
    gradient = compute_gradient(image, mask)
    return absolute_gradient(gradient)


# using computation_method to produce the gradient
def main(computation_method, image, mask):
    return computation_method(image, mask)


# constant for this stage
ALL_METHODS = [compute_gradient, compute_absolute_gradient]

METHOD_TO_NAME = {
    compute_gradient: "Normal_Gradient",
    compute_absolute_gradient: "Absolute_Gradient"
}
