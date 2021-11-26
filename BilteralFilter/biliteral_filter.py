import cv2
import numpy as np
import matplotlib.pyplot as plt


figsize = (10, 10)

def gaussian(x, sigma, mu=0):
    """ get gaussin function """
    
    return np.exp(- (x ** 2) / (2.0 * (sigma ** 2))) 
    
def distance(x, y, i, j):
    """ get eculidien distance  """
    
    return ((x - i) ** 2 + (y - j) ** 2)  # ** (0.5)

def bilateral_one_pixel(source, x, y, radius, sigma_r, sigma_s):
    """ run bilateral filter on a single pixel of the image """
    
    # === init vars
    filtered_pix = 0
    UpperBound = 0
    Wp = 0

    radius = radius.astype(int)
    for i in range(-radius, radius + 1, 1):
        for j in range(-radius, radius + 1, 1):
            if 0 < x + i < source.shape[0] and 0 < y + j < source.shape[1]:
                wp_result = np.exp( - (distance(x, y, x + i, y + j)) / (2.0 * (sigma_s ** 2))) * gaussian(source[x, y] - source[x + i, y + j], sigma_r)
                Wp += wp_result
                UpperBound += source[x + i, y + j] * wp_result

    filtered_pix = UpperBound / Wp
    
    # make result uint8
    filtered_pix = np.clip(filtered_pix, 0, 255).astype(np.uint8)
    return filtered_pix

def bilateral_filter(source, d, sigma_r, sigma_s):
    """ biliteral filter: smooth using guessian and multiple by the distance """
    
    # build empty filtered_image
    filtered_image = np.zeros(source.shape,np.uint8)  
    # make input float 
    source = source.astype(float)
    # d must be odd!
    d = np.floor(d / 2) + 1
    
    # run on all pixels with bilateral_one_pixel
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            filtered_image[i, j] = bilateral_one_pixel(source, i, j, d, sigma_r, sigma_s)
            
    return filtered_image

def upload_noisy_image(src):
    """ show a noisy image """
    
    plt.figure(figsize=(10,10))
    plt.imshow(src,cmap="gray",vmin=0,vmax=255)
    plt.colorbar()
    plt.show()

def run_bilateral_filter(src, d, sigma_r, sigma_s):
    """ run bilateral_filter """
    
    filtered_image = bilateral_filter(src, d, sigma_r, sigma_s)
    show_image(filtered_image, "my biliteral filter")
    
    return filtered_image


def compare_to_cv(src, d, sigma_r, sigma_s):
    """ compare to opencv """
    
    filtered_image_OpenCV = cv2.bilateralFilter(src, d, sigma_r, sigma_s)
    show_image(filtered_image_OpenCV, "cv2 biliteral filter")

def compare_to_gaussian_blur(gauss_noise_im, d, sigma_s):
    """ compare to regular gaussian blur """
    
    blur = cv2.GaussianBlur(gauss_noise_im,(d,d),sigma_s)
    show_image(blur, "gaussian blur image")
    
    return blur

def compare_to_canny(filtered_image, gauss_noise_im):
    """ copare canny results between the two images """
    
    th_low = 100
    th_high = 200
    res = cv2.Canny(filtered_image,th_low,th_high)
    show_image(res, "cv2 Canny image on filtered image")
    
    res = cv2.Canny(gauss_noise_im, th_low, th_high)
    show_image(res, "cv2 Canny image on gaessisan noise")


def show_image(img, text, figsize=(10,10)):
    """ show the image given """
    
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.colorbar()
    plt.title(text)
    plt.show()
    

def main():
    src = cv2.imread("butterfly_noisy.jpg")
    src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    d = 5        # edge size of neighborhood perimeter 
    sigma_r = 12 # sigma range
    sigma_s = 16 # sigma spatial
    
    upload_noisy_image(src)
    filtered_image = run_bilateral_filter(src, d, sigma_r, sigma_s)
    compare_to_cv(src, d, sigma_r, sigma_s)
    gauss_noise_im = compare_to_gaussian_blur(src, d, sigma_s)
    compare_to_canny(filtered_image, gauss_noise_im)


if __name__ == '__main__':
    main()
