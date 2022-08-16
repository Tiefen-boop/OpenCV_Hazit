import copy

import numpy as np

from helpFunctions import *


# Reading the required image in
# which operations are to be done.
# Make sure that the image is in the same
# directory in which this python program is


def continue_hough_space(images):
    # lines=helpFunctions.findMaxValuedLines(hough_space,2)
    # print(lines)
    titles = ['original image', 'gray', 'laplaced', 'masked', 'hough space']
    # images = [img, gray, laplaced, filtered, masked, hough_space]
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


#    with open("hough_space.txt") as textFile:
#        hough_spaceAfterReload = [line.split() for line in textFile]
#    hough_spaceAfterReload=np.array(hough_spaceAfterReload,dtype=int)
#    images=[hough_space,hough_spaceAfterReload]
#    titles = ['hough_space', 'hough_spaceAfterReload']
#    for i in range(len(images)):
#        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
#        plt.title(titles[i])
#        plt.xticks([]), plt.yticks([])
#    plt.show()

def main2(image, laplacians, hough_spaces):
    drawn_images = np.zeros(hough_spaces.shape, dtype=object)
    scoring_methods = [score_by_frequency2, score_by_frequency]
    for i in range(len(laplacians)):
        for j in range(len(hough_spaces[i])):
            lines = find_max_valued_lines(hough_spaces[i][j], laplacians[i], 20)
            drawn_images[i][j] = [copy.deepcopy(image) for i in range(len(scoring_methods))]
            for k in range(len(scoring_methods)):
                top_lines = get_top_lines_2(lines, laplacians[i], scoring_methods[k])
                top_lines_cut_to_intersection = cut_to_intersection(top_lines)
                draw_all_lines(drawn_images[i][j][k], top_lines_cut_to_intersection)
    plottings = [[image, laplacians[i],  drawn_images[i][0][0], drawn_images[i][0][1], drawn_images[i][1][0], drawn_images[i][1][1]] for i in range(len(laplacians))]
    plot_images(plottings[0], ['original', 'laplaces (normal)', 'O(n^2) by density', 'O(n^2) by quality', 'O(n) by density', 'O(n) by quality'])
    plot_images(plottings[1], ['original', 'laplaces (with abs)', 'O(n^2) by density', 'O(n^2) by quality', 'O(n) by density', 'O(n) by quality'])


def main(image, laplaced, hough_space):
    titles = ['original image', 'hough_space', 'laplaced']
    images = [image, hough_space, laplaced]
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    lines = find_max_valued_lines(hough_space, laplaced, 20)
    lines_scored = np.array([[line, score_by_density(line, laplaced)] for line in lines], dtype=object)
    indices_of_top4 = np.argpartition(lines_scored[:, 1], 0)[-6:]
    top4 = get_top_lines_2(lines, laplaced, score_by_frequency)
    #cut_to_intersection(top4)
    draw_all_lines(image, top4)
    titles = ['drawn image', 'hough_space', 'laplaced']
    images = [image, hough_space, laplaced]
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()



if __name__ == '__main__':
    with open("hough_space.txt") as textFile:
        hough_space = [line.split() for line in textFile]
    hough_space = np.array(hough_space, dtype=int)
    imgAddress = "01563.png"
    img = cv2.imread(imgAddress)
    with open("laplaced.txt") as textFile:
        laplaced = [line.split() for line in textFile]
    laplaced = np.array(laplaced, dtype=int)
    main(img, laplaced, hough_space)
