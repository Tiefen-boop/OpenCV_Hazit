import datetime
import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt


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


def plot_images(images, titles, show=True, dir_to_save=None):
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    # plt.savefig('plots/' + str(datetime.today()).split('.')[0].replace(':', '-') + '.png')
    if dir_to_save is None:
        os.makedirs("plots", exist_ok=True)
        plt.savefig('plots/' + str(datetime.today()).replace('.', '-').replace(':', '-') + '.png')
    else:
        os.makedirs(dir_to_save, exist_ok=True)
        plt.savefig(dir_to_save + "/" + str(datetime.today()).replace('.', '-').replace(':', '-') + '.png')
    if show:
        plt.show()


def find_index_of_array_value_in_array(array, value):
    for i in range(len(array)):
        if np.array_equal(array[i], value):
            return i
    return -1


def edged_line(line, edges):  # edges is an array of 2 indices
    flat_list = [item for sublist in edges for item in sublist]
    min_intersection_index = min(flat_list)
    max_intersection_index = max(flat_list)
    return np.array(line[min_intersection_index:max_intersection_index + 1])
