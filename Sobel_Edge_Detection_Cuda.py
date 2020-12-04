import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarr
import numpy as np
from pycuda.compiler import SourceModule

def sobel(img):
    h, w = img.shape

    #this will calculate the horizontal gradient ie - the vertical edges
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.int32)
    horizontal_cuda = gpuarr.to_gpu(horizontal)

    #this will calculate the vertical gradient ie - the horizontal edges
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.int32)
    vertical_cuda = gpuarr.to_gpu(vertical)

    img_cuda = gpuarr.to_gpu(img.astype(np.int32))

    result_cuda = gpuarr.zeros((h, w), np.int32)
    result = np.zeros((h, w), np.int32)

    module = SourceModule("""
        __global__ void convolve(int horizontal[3][3], int vertical[3][3], int img[480][640], int result[480][640])
        {
            int i = blockIdx.x + 1;
            int j = blockIdx.y + 1;
            
            int horizontalDiff = 0;
            int verticalDiff = 0;

            for(int k = -1; k <= 1; k++)
            {
                for(int l = -1; l <=1; l++)
                {
                    horizontalDiff += horizontal[1+k][1+l] * img[i + k][j + l];
                    verticalDiff += vertical[1+k][1+l] * img[i + k][j + l];
                }
            }
            result[i-1][j-1] = sqrt((float)(horizontalDiff * horizontalDiff + verticalDiff * verticalDiff));
        }
    """)

    func = module.get_function("convolve")
    func(horizontal_cuda, vertical_cuda, img_cuda, result_cuda, block = (1, 1, 1), grid = (478, 638))
    result = result_cuda.get()
    return result.astype(np.uint8)


    """
    for k in range (-1, 1):
                for l in range (-1, 1):
                    horizontalDiff += horizontal[1 + k][1 + l] * src_img[i + k][j + l]
                    verticalDiff += vertical[1 + k][1 + l] * src_img[i + k][j + l]
            gradient_image[i-1][j-1] = np.sqrt(pow(horizontalDiff, 2.0) + pow(verticalDiff, 2.0))
    """