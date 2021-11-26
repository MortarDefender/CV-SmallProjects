import cv2
import numpy as np
import matplotlib.pyplot as plt


def set_img_to_binary_color(im_gray):
    """ convert the colored image to a black or white colors only """
    
    im_th = cv2.threshold(im_gray.copy(), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    show_image(im_th, "binary image")
    
    return im_th


def pixel_merge(im_th):
    """ merge all pixels of the same word together to make one connected component using a morphologic operator """
    
    kernel = np.zeros((5,5),dtype=np.uint8)
    kernel[2,:] = 1
    kernel[:,2] = 1
    dilated_im = cv2.dilate(im_th, kernel, iterations = 1)
    show_image(dilated_im, "dilated image")
    
    return dilated_im, kernel


def find_words(dilated_im, im):
    """ find words using connectedComponentsWithStats in the im given """
    
    res = im.copy()

    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(dilated_im.copy(), 4, cv2.CV_32S)

    for i in range(numLabels):
        componentMask = (labels == i).astype("uint8") * 255
        plot_rec(componentMask, res)

    return res


def plot_rec(mask, res_im):
    """ plot a rectengle around each word in res image using mask image of the word """
    
    xy = np.nonzero(mask)
    y = xy[0]
    x = xy[1]
    left = x.min()
    right = x.max()
    up = y.min()
    down = y.max()

    res_im = cv2.rectangle(res_im, (left, up), (right, down), (0, 20, 200), 2)
    return res_im


def show_word_finder(dilated_im, im):
    """ show the image after calling the find words function """
    
    show_image(find_words(dilated_im,im), "find words")


def mark_title_words(dilated_im, im, kernel):
    """ dilate the im given and call find words function """
    
    d2 = cv2.dilate(dilated_im, kernel, iterations = 1)
    binary_only_title_cc_img = cv2.erode(d2, kernel, iterations = 1)
    show_image(find_words(binary_only_title_cc_img,im), "find title words")


def show_image(img, title, figsize=(20, 20)):
    """ show the image given """
    
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.show()


def show_grey_img(im_gray, title, figsize=(10, 10)):
    """ show the image given in grey scale """
    
    plt.figure(figsize=figsize)
    plt.imshow(im_gray,cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def main():
    im = cv2.imread("news.jpg")
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    show_grey_img(im_gray, "original image in grey scale")
    im_th = set_img_to_binary_color(im_gray)
    dilated_im, kernel = pixel_merge(im_th)

    show_word_finder(dilated_im, im)
    mark_title_words(dilated_im, im, kernel)


if __name__ == '__main__':
    main()
