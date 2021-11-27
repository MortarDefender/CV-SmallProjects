import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_original_image():
    """ show the original image and return it and a gray scale of it """
    original_image = cv2.imread("circles.bmp")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    plot_image(original_image, "original image", True)
    
    return original_image, grey_image


def hugh_transform_circle(original_image):
    """ find the circles in the image using canny for edge detenction, and the eq for circle in space """
    
    edgeImage = cv2.Canny(original_image, 50, 400)
    
    plot_image(edgeImage, "edge image")
    maxRadius = 25

    radiusStep = 1
    radius = np.arange(-maxRadius, maxRadius, radiusStep)

    firstPointStep = 1
    firstPoint = np.arange(0, original_image.shape[0], firstPointStep)
    
    secondPointStep = 1
    secondPoint = np.arange(0, original_image.shape[1], secondPointStep)
    acc_mat = np.zeros((secondPoint.shape[0], firstPoint.shape[0], radius.shape[0]))
    
    # ## Fill accumulation matrix
    edge_inds = np.argwhere(edgeImage > 0)
    
    # run on all a, b and edge indices and find corresponding R
    for yx in edge_inds:
        x = yx[1]
        y = yx[0]
        print("running on edge: {} ...".format(yx))
        
        for a_ind, a0 in enumerate(firstPoint):
            for b_ind, b0 in enumerate(secondPoint):
    
                r0 = np.sqrt((a0 - x) ** 2 + (b0 - y) ** 2)

                if r0 > maxRadius:
                    continue
    
                r_ind = np.argmin(np.abs(r0 - radius))
                acc_mat[b_ind, a_ind, r_ind] += 1
    
    plot_image_mat(acc_mat, firstPoint, secondPoint, 'accumulation matrix maxed over r axis')

    # ## Threshold accumulation matrix
    TH = 25
    acc_mat_th = acc_mat > TH
    
    plot_image_mat(acc_mat_th, firstPoint, secondPoint, 'accumulation matrix TH summed over r axis')

    # ## Min distance
    edge_inds = np.argwhere(acc_mat_th > 0)
    
    min_dist = 15
    
    acc_mat_th_dist = acc_mat_th.copy()

    # run on all above TH bins
    for i in range(edge_inds.shape[0]):
        b0, a0, r0 = edge_inds[i]
    
        # search in all other above TH bins
        for j in range(i + 1, edge_inds.shape[0]):
            b1, a1, r1 = edge_inds[j]
            
            # if the two above are neighbors (below min_dist) - delete the less important
            if ((r0 - r1) * radiusStep) ** 2 + ((a0 - a1) * firstPointStep) ** 2 + ((b0 - b1) * secondPointStep) ** 2 < min_dist ** 2:
                if acc_mat[b0, a0, r0] >= acc_mat[b1, a1, r1]:
                    acc_mat_th_dist[b1, a1, r1] = 0
                else:
                    acc_mat_th_dist[b0, a0, r0] = 0
    
    plot_image_mat(acc_mat_th_dist, firstPoint, secondPoint, 'accumulation matrix TH and min_dist summed over r axis')
    plot_circle(original_image, acc_mat_th_dist, radius, firstPoint, secondPoint)
    

def plot_circle(original_image, acc_mat_th_dist, r, a, b):
    """ Plot circles found by hough """
    
    edge_inds = np.argwhere(acc_mat_th_dist > 0)
    
    res = original_image.copy()
    for b_ind, a_ind, r_ind in edge_inds:
        r0 = r[r_ind]
        a0 = a[a_ind]
        b0 = b[b_ind]
        
        # draw the outer circle
        res = cv2.circle(res, (a0, b0), r0, (0, 255, 0), 1)
    
    plot_image(res, "final result")


def compare_to_cv(original_image, grey_image):
    """ compare my impolementation to cv2 hugh circles impolementation """
    
    res = original_image.copy()
    circles = cv2.HoughCircles(grey_image, cv2.HOUGH_GRADIENT, 1,
                               10, param1=100, param2=8, minRadius=5, maxRadius=30)
    
    for xyr in circles[0, :]:
        # draw the outer circle
        res = cv2.circle(res, (xyr[0], xyr[1]), xyr[2], (0, 255, 0), 1)
    
    plot_image(res, "final result- cv2.HoughCircles")


def plot_image_mat(acc_mat, a, b, text):
    """ plot an image matrix """
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.sum(acc_mat, axis = 2), extent=[b.min(), b.max(), a.max(), a.min()], aspect = 'auto')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title(text)
    plt.colorbar()
    plt.show()


def plot_image(res, text, gray=False, figsize=(10, 10)):
    """ plot an image """
    
    plt.figure(figsize=figsize)
    
    if gray:
        plt.imshow(res, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(res)
        
    plt.title(text)
    plt.show()


def main():
    original_image, grey_image = show_original_image()
    hugh_transform_circle(original_image)
    compare_to_cv(original_image, grey_image)
    

if __name__ == '__main__':
    main()
