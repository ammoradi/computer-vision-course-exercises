import numpy as np
import cv2 as cv

# Read the image
img = cv.imread("assets/img/rectangles.jpg", 0)
 
# Thresholding the image
(thresh, img_bin) = cv.threshold(img, 128, 255,cv.THRESH_BINARY|cv.THRESH_OTSU)

# Invert the image
img_bin = 255-img_bin 

#Create default parametrization LSD
lsd = cv.createLineSegmentDetector(0)

#Detect lines in the image
lines = lsd.detect(img_bin)[0] #Position 0 of the returned tuple are the detected lines

#Draw detected lines in the image
output = lsd.drawSegments(img_bin,lines)

#Show image
cv.imshow("LSD",output)
cv.waitKey(0)