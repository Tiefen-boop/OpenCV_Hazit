import numpy as np
import sympy as sy


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


def is_line_unique(image, line, lines, max_distance=6):
    r, theta = cart_to_polar(line)
    for existing_line in lines:
        r2, theta2 = cart_to_polar(existing_line)
        if round(theta, 3) == round(theta2, 3) and abs(r - r2) < max_distance:
            return False
    return True


def is_line_unique_by_alpha(image, line, lines, max_diff_alpha=25, max_distance=25):
    r, theta = cart_to_polar(line)
    alpha = get_alpha_by_theta(theta)
    for existing_line in lines:
        r2, theta2 = cart_to_polar(existing_line)
        alpha2 = get_alpha_by_theta(theta2)
        if abs((90 - abs(alpha)) - (90 - abs(alpha2))) < max_diff_alpha and (
        not is_line_unique_by_avg_distance(image, line, [existing_line], max_distance)):
            return False

    return True


def get_alpha_by_theta(theta):
    if theta == 0:  # line is of the form X=const
        return 90
    if theta == 90:  # line is of the form y=const
        return 0
    if theta > 0:
        alpha = theta - 90
    else:
        alpha = theta + 90
    return alpha


def get_alpha_of_line(line):
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[-1][0]
    y2 = line[-1][1]
    m = (y2 - y1) / (x2 - x1)
    coefs = [m, y1 - m * x1]  # returns [c1, c0] for y=c1*x + c0
    # coefs = np.polyfit([x1, x2], [y1, y2], 1)
    r = int(np.abs(coefs[1]) / np.sqrt((coefs[0] * coefs[0]) + 1))
    alpha = int(np.arctan(coefs[0]) * 180 / np.pi)


def line_to_linear_equation_function_x_to_fx(line):
    x1, y1 = line[0][:2]
    x2, y2 = line[-1][:2]
    if x1 == x2:
        raise Exception(
            "Line is of the form x=const, use the function " + line_to_inverse_linear_equation_function_y_to_x.__name__)
        # return lambda x: x1
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b


def line_to_inverse_linear_equation_function_y_to_x(line):
    y1, x1 = line[0][:2]
    y2, x2 = line[-1][:2]
    if x1 == x2:
        raise Exception(
            "Line is of the form y=const, use the function " + line_to_linear_equation_function_x_to_fx.__name__)
        # return lambda x: x1
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b


"""
this function is implemented by the claim that the average distance is  derived from integral value.
"""


# not correct, maybe can be modified tp: if more than N "x" values have a distance lower than max_distance,
# then the line is not unique
def is_line_unique_by_distance_for_each_x(image, line, lines, max_distance=6):
    first_line_func = line_to_linear_equation_function_x_to_fx(line)
    for existing_line in lines:
        second_line_func = line_to_linear_equation_function_x_to_fx(existing_line)
        for x in range(line[0][0], line[-1][0]):
            y1 = first_line_func(x)
            y2 = second_line_func(x)
            if abs(y1 - y2) < max_distance:
                return False
    return True


def is_line_unique_by_avg_distance(image, line, lines, max_distance=6):
    x_start = min(line[0][0], line[-1][0])
    x_end = max(line[0][0], line[-1][0])
    y_start = min(line[0][1], line[-1][1])
    y_end = max(line[0][1], line[-1][1])

    inverse = False

    if (y_start == 0 or y_start == 1) and y_end == len(image) - 1:
        first_line_func = line_to_inverse_linear_equation_function_y_to_x(line)
        inverse = True
    else:
        first_line_func = line_to_linear_equation_function_x_to_fx(line)
    for existing_line in lines:
        sum_of_distances = 0
        if inverse:
            try:
                second_line_func = line_to_inverse_linear_equation_function_y_to_x(existing_line)
            except Exception as e:
                continue  # the lines are orthogonal, so they are unique
        else:
            try:
                second_line_func = line_to_linear_equation_function_x_to_fx(existing_line)
            except Exception as e:
                continue  # the lines are orthogonal, so they are unique

        if inverse:  # line is of the form x=const
            y_start = min(line[0][1], line[-1][1])
            y_end = max(line[0][1], line[-1][1])
            for y in range(y_start, y_end):
                sum_of_distances += abs(first_line_func(y) - second_line_func(y))
            average_distance = sum_of_distances / (y_end - y_start)
        else:
            for x in range(x_start, x_end + 1):
                y1 = first_line_func(x)
                y2 = second_line_func(x)
                sum_of_distances += abs(y1 - y2)
            average_distance = sum_of_distances / (x_end - x_start)
        if average_distance < max_distance:
            return False
    return True


def is_line_unique_by_avg_distance_using_integral(image, line, lines, max_distance=6):
    x_start = min(line[0][0], line[-1][0])
    x_end = max(line[0][0], line[-1][0])
    inverse = False
    if x_start == x_end:
        first_line_func = line_to_inverse_linear_equation_function_y_to_x(line)
        inverse = True
    else:
        first_line_func = line_to_linear_equation_function_x_to_fx(line)
    for existing_line in lines:
        if inverse:
            try:
                second_line_func = line_to_inverse_linear_equation_function_y_to_x(existing_line)
            except Exception as e:
                continue  # the lines are orthogonal, so they are unique
        else:
            try:
                second_line_func = line_to_linear_equation_function_x_to_fx(existing_line)
            except Exception as e:
                continue  # the lines are orthogonal, so they are unique
        if inverse:  # line is of the form x=const
            y_start = min(line[0][1], line[-1][1])
            y_end = max(line[0][1], line[-1][1])
            y = sy.Symbol('y')
            integral_res = sy.integrate(abs(x_start - second_line_func(y)), (y, y_start, y_end))
            average_distance = integral_res / abs(y_end - y_start)
        else:
            x = sy.Symbol('x')
            integral_res = sy.integrate(first_line_func(x) - second_line_func(x), (x, x_start, x_end))
            average_distance = integral_res / abs(x_end - x_start)
        if average_distance < max_distance:
            return False
    return True


ALL_METHODS = [is_line_unique_by_avg_distance_using_integral, is_line_unique_by_avg_distance, is_line_unique_by_alpha,
  is_line_unique]
# ALL_METHODS = [is_line_unique_by_alpha]

METHOD_TO_NAME = {
    is_line_unique_by_avg_distance_using_integral: "uniqueness by avg distance using integral",
    is_line_unique_by_avg_distance: "uniqueness by avg distance",
    is_line_unique_by_alpha: "uniqueness by alpha",
    is_line_unique: "uniqueness by r, theta"

}
