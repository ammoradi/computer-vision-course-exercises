import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# Source image
src = cv.imread('assets/img/striping.bmp')
# Destination image
# to be a same dimension image, its load the source initially
dst = cv.imread('assets/img/striping.bmp')
# Denoise the image
cv.fastNlMeansDenoising(src,dst,150.0,21,7)
# Show result
plt.subplot(121),plt.imshow(src)
plt.subplot(122),plt.imshow(dst)
plt.show()