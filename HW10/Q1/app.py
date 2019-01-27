import cv2 as cv
import numpy as np


def resize(img, height):
    (h, w) = img.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
img_num = "4"
image = cv.imread("./Images/" + img_num + ".jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()

image = resize(image, 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
print("STEP 1: Edge Detection")
edged = cv.Canny(gray, 75, 200)

# show the original image and the edge detected image
cv.imshow("Image", image)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("Edged", edged)
cv.waitKey(0)
cv.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
print("STEP 2: Find contours of paper")
cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[len(cnts) - 2]
cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

# show the contour (outline) of the piece of paper
cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv.imshow("Outline", image)
cv.waitKey(0)
cv.destroyAllWindows()

# prepare four coordinates shape
pts = screenCnt.reshape(4, 2) * ratio
rect = np.zeros((4, 2), dtype="float32")

# coordinates: [ top-left, top-right, bottom-right, bottom-left]

# top-left point -> smallest sum
# bottom-right point -> largest sum
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

# top-right point -> smallest difference,
# bottom-left ->largest difference
diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

# define four point rectangle of coordinates
(tl, tr, br, bl) = rect

# top width = distance(top-left, top-right)
# bottom width = distance(bottom-left, bottom-right)
# rectangle width = max(top width, botton width)
topWidth = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
bottomWidth = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
maxWidth = max(int(topWidth), int(bottomWidth))

# right height = distance(top-right, bottom-right)
# left height = distance(top-left, bottom-left)
# rectangle width = max(right height, left height)
rightHeight = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
leftHeight = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(rightHeight), int(leftHeight))

# get top-down view of calculated rectangle
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")

# compute the perspective transform matrix and then apply it
print("STEP 3: Apply perspective transform")
M = cv.getPerspectiveTransform(rect, dst)
warped2 = cv.warpPerspective(orig, M, (maxWidth, maxHeight))

# if you want thresholded gray scale output (like a scanner) uncomment lines below.
# warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
# T = cv.adaptiveThreshold(warped, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 0)

# show the original and scanned images
result = np.concatenate((resize(orig, height=300), resize(warped2, height=300)), axis=1)
cv.imwrite("./Outputs/" + img_num + ".jpg", result)
cv.imshow("Result", result)
cv.waitKey(0)
