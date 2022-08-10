from helpFunctions import *


# Reading the required image in
# which operations are to be done.
# Make sure that the image is in the same
# directory in which this python program is


def continue_hough_space(images):
    # lines=helpFunctions.findMaxValuedLines(hough_space,2)
    # print(lines)
    titles = ['original image', 'gray', 'laplaced', 'filtered', 'masked', 'hough space']
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

def main(image ,laplaced,hough_space):
    titles = ['original image', 'hough_space', 'laplaced']
    #this is neccesery beacuse we didnt save the filtered laplace in
    with open("laplaced.txt") as textFile:
        laplaced = [line.split() for line in textFile]
    laplaced = np.array(laplaced, dtype=int)
    images = [image, hough_space, laplaced]
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    lines = find_max_valued_lines(hough_space, laplaced, 20)
    lines_scored = np.array([[line, score_by_gradients_quality(line)] for line in lines], dtype=object)
    indices_of_top4 = np.argpartition(lines_scored[:, 1], 0)[-6:]
    top4 = lines_scored[indices_of_top4][:, 0]
    draw_all_lines(image, top4)


if __name__ == '__main__':
    with open("hough_space.txt") as textFile:
        hough_space = [line.split() for line in textFile]
    hough_space = np.array(hough_space, dtype=int)
    imgAddress = "images/imagesForTesting/988Cropped.jpg"
    img = cv2.imread(imgAddress)
    with open("laplaced.txt") as textFile:
        laplaced = [line.split() for line in textFile]
    laplaced = np.array(laplaced, dtype=int)
    main(img, laplaced, hough_space)
