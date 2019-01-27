#!/usr/bin/env python3
import cv2 as cv
import numpy as np

if __name__ == '__main__':

    # read input image
    img_name = 'right04'
    img_bgr = cv.imread('./Images/' + img_name + '.jpg')

    # convert colored image to gray scale image
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # convert gray scale image to numpy float32 array
    img_gray = np.float32(img_gray)

    # calculate vertical and horizontal derivation of image
    # using sobel filter with 3x3 kernel
    Ix, Iy = cv.Sobel(img_gray, cv.CV_32F, 1, 0, ksize=3),\
             cv.Sobel(img_gray, cv.CV_32F, 0, 1, ksize=3)
    cv.imwrite('./Outputs/Ix-' + img_name + '.jpg', Ix)
    cv.imwrite('./Outputs/Iy-' + img_name + '.jpg', Iy)

    # calculate the square of derivations
    Ix_square, Iy_square, Ix_dot_Iy = Ix ** 2, Iy ** 2, Ix * Iy
    cv.imwrite('./Outputs/Ix_square-' + img_name + '.jpg', Ix_square)
    cv.imwrite('./Outputs/Iy_square-' + img_name + '.jpg', Iy_square)
    cv.imwrite('./Outputs/Ix_dot_Iy-' + img_name + '.jpg', Ix_dot_Iy)

    # apply 3x3 gaussian blur on derivation's squares
    g_Ix_square, g_Iy_square, g_Ix_dot_Iy = cv.GaussianBlur(Ix_square, (3, 3), 0),\
                                            cv.GaussianBlur(Iy_square, (3, 3), 0),\
                                            cv.GaussianBlur(Ix_dot_Iy, (3, 3), 0)
    cv.imwrite('./Outputs/g_Ix_square-' + img_name + '.jpg', g_Ix_square)
    cv.imwrite('./Outputs/g_Iy_square-' + img_name + '.jpg', g_Iy_square)
    cv.imwrite('./Outputs/g_Ix_dot_Iy-' + img_name + '.jpg', g_Ix_dot_Iy)

    # initializing parameters of below formula
    # R = det(M) - k(trace(M)) ** 2
    k = 0.04
    R = np.zeros(img_gray.shape, dtype=float)

    # calculate R by loop iteration
    for i in range(len(img_gray)):
        for j in range(len(img_gray[i])):
            M = np.array(
                [
                    [g_Ix_square[i][j], g_Iy_square[i][j]],
                    [g_Ix_dot_Iy[i][j], g_Iy_square[i][j]]
                ]
            )
            R[i][j] = np.linalg.det(M) - k * np.trace(M) ** 2

    # threshold value of R to detect corners and draw it on input image
    for i in range(len(img_gray)):
        for j in range(len(img_gray[i])):
            if R[i][j] > 0.01 * R.max():
                img_bgr[i][j] = [0, 0, 255]

    cv.imshow('output', img_bgr)
    cv.imwrite('./Outputs/OUTPUT-' + img_name + '.jpg', img_bgr)
    cv.waitKey(0)
