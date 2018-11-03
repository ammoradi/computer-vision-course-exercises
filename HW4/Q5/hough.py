import numpy as np
import cv2 as cv

# load the image and then convert it to grayscale
img = cv.imread("assets/img/rectangles.jpg")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray,100,200,apertureSize = 3)

minLineLength = 30
maxLineGap = 10
lines = cv.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

#Show images
cv.imshow('HOUGH',img)
cv.imshow('RECTS',edges)
cv.waitKey(0)
cv.destroyAllWindows()