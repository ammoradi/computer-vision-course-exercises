import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# get a grayscale image as input.
# calculate the final sigma for otsu algorithm (can be calculated by two approches).
# apply binarification to image based on final sigma.
def otsu(img):
    total_pixels_number = img.shape[0] * img.shape[1]
    m_weight = 1.0/total_pixels_number #mean weight
    his, bins = np.histogram(img, np.array(range(0, 256)))
    final_index = -1
    final_sigW = -1
    for i in bins[1:-1]:
		# calculate weights
        W1 = np.sum(his[:i]) * m_weight
        W2 = np.sum(his[i:]) * m_weight
		# calculate sigmas
        sig1 = np.mean(his[:i])
        sig2 = np.mean(his[i:])

        # sigW = (W1 * (sig1)**2) + (W2 * (sig2)**2) #to use this approach, should change Ws definitions.
        sigW = W1 * W2 * (sig1 - sig2) ** 2 #its better approach

        print("=================")
        print("histogram index ==>", i)
        print("weight of class 1 ==>", W1)
        print("weight of class 2 ==>", W2)
        print("variance of class 1 ==>", sig1)
        print("variance of class 2 ==>", sig2)
        print("total variance ==>", sigW)

        if sigW > final_sigW:
            final_index = i
            final_sigW = sigW

    final_img = img.copy()

    print("\n#################")
    print("### final histogram index:", final_index)
    print("#################")
	# classification
    final_img[img > final_index] = 255
    final_img[img < final_index] = 0
    return final_img

#run part of app
file_name = 'assets/redBall.png'
src = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
otsu_res = otsu(src)
cv.imshow("otsu result", otsu_res)
cv.waitKey(0)
