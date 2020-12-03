import numpy as np          
import cv2 
import convolution_naive as cnv

def blur_and_shapen(img):
    
    h,w=img.shape
    blur_filter=np.ones((3,3))/9
    blur_img=cv2.filter2D(img,-1,blur_filter) # using inbuilt filter for now
    
    # blur_img=cnv.conv_naive(img,blur_filter)
    
    sharpen_filter1=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpen_filter2=np.array([[1,1,1],[1,-7,1],[1,1,1]])  # traditional filter but not as good as filter 1
    sharpened_img1=cv2.filter2D(blur_img,-1,sharpen_filter1)
    sharpened_img2=cv2.filter2D(blur_img,-1,sharpen_filter2)
    
    #sharpened_img1=cnv.conv_naive(blur_img,sharpen_filter1)   this is using naive version
    #sharpened_img2=cnv.conv_naive(blur_img,sharpen_filter2)   naive version
    
    return blur_img,sharpened_img1,sharpened_img2