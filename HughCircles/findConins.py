import cv2
from matplotlib import pyplot as plt


def main():
    figsize = (10, 10)
    
    im3 = cv2.imread("coins.png")
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
    im = cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)
    res = im3.copy()
    
    acc_th = 28
    min_dist = 30
    acc_ratio = 1
    canny_upper_th = 450
    
    circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, acc_ratio,
                               min_dist, param1=canny_upper_th,
                               param2=acc_th, minRadius=40, maxRadius=68)
    
    # for each detected circle
    for xyr in circles[0, :]:
        # draw the outer circle
        res = cv2.circle(res, (xyr[0], xyr[1]), xyr[2], (0, 255, 0), 3)
    
        if xyr[2] <= 53:
            addText(res, "Dime", (xyr[0], xyr[1]))
        elif 55 < xyr[2] <= 61:
            addText(res, "Nickel", (xyr[0], xyr[1]))
        elif 63 < xyr[2] <= 68:
            addText(res, "Quarter", (xyr[0], xyr[1]))
    
    
    plt.figure(figsize=figsize)
    plt.imshow(res)
    plt.title("final result- coins detection")
    plt.show()


def addText(img, text, cor, offset = 40):
    """ add text to the img given in the cord given """

    lineType = 2
    fontScale = 0.8
    fontColor = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(img, text, ((cor[0] - offset).astype(int), cor[1]),
                font, fontScale, fontColor, 2, lineType)
    

if __name__ == '__main__':
    main()
