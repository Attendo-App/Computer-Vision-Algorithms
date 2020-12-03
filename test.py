import numpy as np
import cv2 as cv
import sobeledgedetectionnaive as sobelnaive
import blur_kernel as blr
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
    #edges = sobelnaive.sobel(gray)
    if(x == 0):
        print(type(gray))
        #print(edges)
        x = x + 1
    blur_img,sharp1,sharp2 = blr.blur_and_shapen(gray)
    # Display the resulting frame
    #edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    cv.imshow('original', gray.astype('uint8'))
    #cv.imshow('frame', edges.astype('uint8'))
    cv.imshow('blur', blur_img.astype('uint8'))
    cv.imshow('sharp1', sharp1.astype('uint8'))
    cv.imshow('sharp2', sharp2.astype('uint8'))
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()