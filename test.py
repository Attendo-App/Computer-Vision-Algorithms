import numpy as np
import cv2 as cv
import blur_kernel as blr
import Sobel_Edge_Detect_Naive as sobelNaive
import Sobel_Edge_Detection_Cuda as sobelCuda

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
x = 0 
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    if(x == 0):
        print(w)
        #print(edges)
        x = x + 1
    edges = sobelCuda.sobel(gray)
    
    #blur_img,sharp1,sharp2 = blr.blur_and_shapen(gray)
    # Display the resulting frame

    #cv.imshow('original', gray.astype('uint8'))
    #cv.imshow('frame', edges.astype('uint8'))
    #cv.imshow('blur', blur_img.astype('uint8'))
    #cv.imshow('sharp1', sharp1.astype('uint8'))
    #cv.imshow('sharp2', sharp2.astype('uint8'))

    # Display the resulting frame
    cv.imshow('frame', edges.astype('uint8'))

    #exit on press q
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()