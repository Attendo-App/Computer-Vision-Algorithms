import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

def sobel(img):
    h, w = img.shape

    #h_cuda = cuda.mem_alloc(4)
    #print(type(h_cuda))
    #cuda.memcpy_htod(h_cuda, h)

    #w_cuda = cuda.mem_alloc(4)
    #cuda.memcpy_htod(w_cuda, w)

    #this will calculate the horizontal gradient ie - the vertical edges
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.int16)
    horizontal_cuda = cuda.mem_alloc(horizontal.nbytes)
    cuda.memcpy_htod(horizontal_cuda, horizontal)

    #this will calculate the vertical gradient ie - the horizontal edges
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.int16)
    vertical_cuda = cuda.mem_alloc(vertical.nbytes)
    cuda.memcpy_htod(vertical_cuda, vertical)


    img_cuda = cuda.mem_alloc(img.astype(np.float32).nbytes)
    cuda.memcpy_htod(img_cuda, img.astype(np.float32))

    result = np.zeros((h,w), np.float32)
    result_cuda = cuda.mem_alloc(result.nbytes)

    module = SourceModule("""
        __global__ void convolve(short *horizontal, short *vertical, float *img, float *result)
        {
            int i = threadIdx.x + 1;
            int j = blockIdx.x + 1;
            
            float horizontalDiff = 0;
            float verticalDiff = 0;

            for(int k = -1; k <= 1; k++)
            {
                for(int l = -1; l <=1; l++)
                {
                    horizontalDiff += horizontal[1+k + (1+l)*3] * img[i + k + (j + l) * 478];
                    verticalDiff += vertical[1+k + (1+l)*3] * img[i + k + (j + l) * 478];
                }
            }
            result[i - 1 + (j - 1) * 478] = sqrt(horizontalDiff * horizontalDiff + verticalDiff * verticalDiff);
        }
    """)

    func = module.get_function("convolve")
    func(horizontal_cuda, vertical_cuda, img_cuda, result_cuda, block = (478, 1, 1), grid = (638, 1))
    cuda.memcpy_dtoh(result, result_cuda)

    return result.astype(np.uint8)


    """
    for k in range (-1, 1):
                for l in range (-1, 1):
                    horizontalDiff += horizontal[1 + k][1 + l] * src_img[i + k][j + l]
                    verticalDiff += vertical[1 + k][1 + l] * src_img[i + k][j + l]
            gradient_image[i-1][j-1] = np.sqrt(pow(horizontalDiff, 2.0) + pow(verticalDiff, 2.0))
    """