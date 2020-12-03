import numpy as np                   

def conv_naive(img,kernel): # function for convolving(only for 3*3 kernels)
    
    h,w=img.shape
    h_k,w_k = kernel.shape
    src_img=np.zeros((h+1,w+1)) # padding the image with zeros
    for i in range(1,h+1):
        for j in range(1,w+1):
          src_img[i][j]=img[i-1][j-1]
    final_img=np.zeros((h,w))
    for i in range (1, h+1):
        for j in range (1, w+1):
            for k in range (-1, 1):
                for l in range (-1, 1):
                    final_img[i-1][j-1] += kernel[1 + k][1 + l] * src_img[i + k][j + l] 
    return final_img                     
