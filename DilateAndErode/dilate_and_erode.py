import cv2
import numpy as np
import matplotlib.pyplot as plt


def my_dilate(img, kernel):
    """ reform the img thus that the pixels around the true img will be color the oposite """
    
    img2 = img.copy()
    img2[cv2.filter2D(img, -1, kernel) > 0] = 1
    return img2


def show_my_dilate_img(img, kernel):
    my_dilate_img = my_dilate(img,kernel)
    show_image(my_dilate_img, "my dilate image")
    check_my_dilate(my_dilate_img, img, kernel)


def check_my_dilate(my_dilate_img, img, kernel):
    """ check if my implementation is the same as cv counterpart """
    
    if (checkEquelImages(my_dilate_img, cv2.dilate(img, kernel))):
        print("cv2.dilate & my_dilate are the same!")
    else: 
        print("try again...")


def my_erode(img, kernel):
    """ reform the img thus that the pixels around the true img will be color the same """
    
    img2 = img.copy()
    img2[cv2.filter2D(img, -1, kernel) < np.sum(kernel)] = 0
    return img2


def show_my_erode_img(img, kernel):
    my_erode_img = my_erode(img,kernel)
    show_image(my_erode_img, "my erode image") 
    check_my_erode(my_erode_img, img, kernel)


def check_my_erode(my_erode_img, img, kernel):
    """ check if my implementation is the same as cv counterpart """
    
    if (checkEquelImages(my_erode_img, cv2.erode(img, kernel))):
        print("cv2.erode & my_erode are the same!")
    else: 
        print("try again...")


def show_image(img, title, figsize=(10, 10)):
    """ show the image in grey scale """
    
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.show()


def checkEquelImages(img1, img2):
    """ subtract the pixels from img1 to img2 sums it up and checks if it zero """
    
    return np.sum(cv2.split(cv2.subtract(img1, img2))) == 0.0


def main():
    img = np.zeros((50, 50))
    img[20:30, 20:30] = 1

    kernel = np.zeros((5,5),dtype=np.uint8)
    kernel[2,:] = 1
    kernel[:,2] = 1

    show_image(img, "original image")
    show_image(kernel, "the kernel")
    show_my_dilate_img(img, kernel)
    show_my_erode_img(img, kernel)


if __name__ == '__main__':
    main()
