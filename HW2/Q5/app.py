import cv2 as cv

# [load]
src1 = cv.imread('assets/img/3.jpg')
src2 = cv.imread('assets/img/4.jpg')
# [blend_images] - 50 50
alpha = 0.5 # src1's weight
beta = 0.5 # src2's weight
dst1 = cv.addWeighted(src1, alpha, src2, beta, 0.0)
# [blend_images] - 60 40
alpha = 0.6 # src1's weight
beta = 0.4 # src2's weight
dst2 = cv.addWeighted(src1, alpha, src2, beta, 0.0)
# [blend_images] - 40 60
alpha = 0.4 # src1's weight
beta = 0.6 # src2's weight
dst3 = cv.addWeighted(src1, alpha, src2, beta, 0.0)
# [save]
cv.imwrite('50% 3.jpg - 50% 4.jpg',dst1)
cv.imwrite('60% 3.jpg - 40% 4.jpg',dst2)
cv.imwrite('40% 3.jpg - 60% 4.jpg',dst3)
# [display]
cv.imshow('50% 3.jpg - 50% 4.jpg', dst1)
cv.imshow('60% 3.jpg - 40% 4.jpg', dst2)
cv.imshow('40% 3.jpg - 60% 4.jpg', dst3)
cv.waitKey(0)
# [display]
cv.destroyAllWindows()