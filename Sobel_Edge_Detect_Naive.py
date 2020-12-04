import numpy as np
import cv2
import blur_kernel as blr
def sobel(img):
    

    #gray_img = np.round(0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)

    h, w = img.shape
    horizontal1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # using additional filters for better two sided edge detection
    vertical1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    vertical2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    img, im1, im2 = blr.blur_and_shapen(img)  #  blurring the image to reduce noise spikes

    gradient_image = np.zeros((h,w))
    src_img=np.zeros((h+2,w+2))
    for i in range(1,h+1):
        for j in range(1,w+1):
          src_img[i][j]=img[i-1][j-1]
    for i in range (1, h+1):
        for j in range (1, w+1):
            horizontalDiff1 = 0
            verticalDiff1 = 0
            horizontalDiff2 = 0
            verticalDiff2 = 0
            for k in range (-1, 2):
                for l in range (-1, 2):
                    horizontalDiff1 += horizontal1[1 + k][1 + l] * src_img[i + k][j + l]
                    verticalDiff1 += vertical1[1 + k][1 + l] * src_img[i + k][j + l]
                    horizontalDiff2 += horizontal2[1 + k][1 + l] * src_img[i + k][j + l]
                    verticalDiff2 += vertical2[1 + k][1 + l] * src_img[i + k][j + l]
            gradient_image[i-1][j-1] = np.sqrt(pow(horizontalDiff1, 2.0) + pow(verticalDiff1, 2.0)+pow(horizontalDiff2, 2.0) + pow(verticalDiff2, 2.0))/4
    return gradient_image

