import numpy as np
import pycuda
import cv2 as cv
import Sobel_Edge_Detect_Naive as sobelNaive

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
    edges = sobelNaive.sobel(gray)

    # Display the resulting frame
    cv.imshow('frame', edges.astype('uint8'))

    #exit on press q
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()