import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#read image
img = cv.imread('assets/doc_shadow.png', cv.IMREAD_GRAYSCALE)
#blur image
blurred = cv.medianBlur(img,5)

# normal thresholding
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

# adaptive thresholding
thresh5 = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
thresh6 = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

titles = ['BINARY','BINARY_INV','TOZERO','TOZERO_INV',
'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

images = [thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]

for i in range(6):
    plt.subplot(3,2,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()