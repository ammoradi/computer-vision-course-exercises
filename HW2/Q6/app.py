import cv2 as cv
import numpy as np

# [load]
src1 = cv.imread('assets/img/5.jpg')
src2 = cv.imread('assets/img/6.jpg')
# Homogeneous blur
dst1 = cv.blur(src1, (9, 9))
dst2 = cv.blur(src2, (9, 9))
# Gaussian blur
dst11 = cv.GaussianBlur(src1, (9, 9), 0)
dst22 = cv.GaussianBlur(src2, (9, 9), 0)
# Median blur
dst111 = cv.medianBlur(src1, 9)
dst222 = cv.medianBlur(src2, 9)
# [save]
cv.imwrite('5-blur-smooth.jpg',dst1)
cv.imwrite('6-blur-smooth.jpg',dst2)
cv.imwrite('5-gaussian-blur.jpg',dst11)
cv.imwrite('6-gaussian-blur.jpg',dst22)
cv.imwrite('5-median-3x3-blur.jpg',dst111)
cv.imwrite('6-median-3x3-blur.jpg',dst222)
# [display]
cv.imshow('5.jpg Blur smoothing', dst1)
cv.imshow('6.jpg Blur smoothing', dst2)
cv.imshow('5.jpg Gaussian blur', dst11)
cv.imshow('6.jpg Gaussian blur', dst22)
cv.imshow('5.jpg Median blur', dst111)
cv.imshow('6.jpg Median blur', dst222)
cv.waitKey(0)
# [display]
cv.destroyAllWindows()