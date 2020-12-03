import numpy as np

def sobel(img):
    

    #gray_img = np.round(0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)

    h, w = img.shape
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_image = np.zeros((480, 640))
    src_img=np.zeros((482,642))
    for i in range(1,h+1):
        for j in range(1,w+1):
          src_img[i][j]=img[i-1][j-1]
    for i in range (1, h+1):
        for j in range (1, w+1):
            horizontalDiff = 0
            verticalDiff = 0
            for k in range (-1, 1):
                for l in range (-1, 1):
                    horizontalDiff += horizontal[1 + k][1 + l] * src_img[i + k][j + l]
                    verticalDiff += vertical[1 + k][1 + l] * src_img[i + k][j + l]
            gradient_image[i-1][j-1] = np.sqrt(pow(horizontalDiff, 2.0) + pow(verticalDiff, 2.0))
    return gradient_image

