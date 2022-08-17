import datetime
import os
import threading
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

import Hough_Space_Stage


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

lock = threading.Lock()


def plot_images(images, titles, show=True, dir_to_save=None):
    lock.acquire()
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.title('thresholded lalplacian with: ' + str(Hough_Space_Stage.get_threshold()))
    # plt.savefig('plots/' + str(datetime.today()).split('.')[0].replace(':', '-') + '.png')
    if dir_to_save is None:
        os.makedirs("plots", exist_ok=True)
        plt.savefig('plots/' + str(datetime.today()).replace('.', '-').replace(':', '-') + '.png')
    else:
        os.makedirs(dir_to_save, exist_ok=True)
        plt.savefig(dir_to_save + "/" + str(datetime.today()).replace('.', '-').replace(':', '-') + '.png')
    if show:
        plt.show()
    lock.release()


def find_index_of_array_value_in_array(array, value):
    for i in range(len(array)):
        if np.array_equal(array[i], value):
            return i
    return -1


def cut_to_intersection(lines):
    edges = [[] for i in range(len(lines))]
    for first_line_index in range(len(lines)):
        line = lines[first_line_index]

        intersection_indexs = []
        for second_line_index in range(first_line_index + 1, len(lines)):
            line2 = lines[second_line_index]

            lineSet = set([tuple(x) for x in line])
            line2Set = set([tuple(x) for x in line2])
            intersectionValues = np.array([x for x in lineSet & line2Set])

            if intersectionValues.size > 0:
                intersection_index_line1 = find_index_of_array_value_in_array(line, intersectionValues[0])

                edges[first_line_index].append(intersection_index_line1)
                intersection_index_line2 = find_index_of_array_value_in_array(line2, intersectionValues[0])
                edges[second_line_index].append(intersection_index_line2)

    for i in range(len(edges)):
        if not edges[i]:
            edges[i].append(0)
            edges[i].append(lines[i].size - 2)

    return np.array([edged_line(lines[i], edges[i]) for i in range(len(lines))], dtype=object)


def edged_line(line, edges):  # edges is an array of 2 indices
    # flat_list = [item for sublist in edges for item in sublist]
    min_intersection_index = min(edges)
    max_intersection_index = max(edges)
    if min_intersection_index == max_intersection_index:
        return line
    return np.array(line[min_intersection_index:max_intersection_index + 1])


