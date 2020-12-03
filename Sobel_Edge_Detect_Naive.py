import numpy as np

def sobel(img):
    h, w = img.shape

    #this will calculate the horizontal gradient ie - the vertical edges
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    #this will calculate the vertical gradient ie - the horizontal edges
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    #this will be the final result
    gradient_image = np.zeros((h, w))

    #convolution loop
    for i in range (1, h - 1):
        for j in range (1, w - 1):
            horizontalDiff = 0
            verticalDiff = 0
            for k in range (-1, 1):
                for l in range (-1, 1):
                    horizontalDiff += horizontal[1 + k][1 + l] * img[i + k][j + l]
                    verticalDiff += vertical[1 + k][1 + l] * img[i + k][j + l]
            gradient_image[i-1][j-1] = np.sqrt(pow(horizontalDiff, 2.0) + pow(verticalDiff, 2.0))
            
    return gradient_image

