import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('assets/img/1.png',0)
equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('4-res.png',res)

plt.hist(res.ravel(),256,[0,256])
plt.show()