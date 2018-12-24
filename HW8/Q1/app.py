# import the necessary packages
import numpy as np
import utils
import glob
import cv2
import time

start_time = time.time()
# load the template image, convert it to gray scale, and detect edges
# then make template smaller for better performance
template = cv2.imread("Images/Template/template.jpg")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
template = utils.resize(template, width=int(template.shape[1] * 0.5))
(tH, tW) = template.shape[:2]

imageIndex = 0
# loop over the images to find the template in
for imagePath in glob.glob("Images/Flags/*.jpg"):
    imageIndex += 1
    # load the image, convert it to gray scale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    cloneIndex = 0
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        cloneIndex += 1
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = utils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

        # draw a bounding box around the detected region
        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                      (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        filename = "Images/Result/%d/iterate-%d.jpg" % (imageIndex, cloneIndex)
        cv2.imwrite(filename, clone)
        # cv2.imshow("Visualize", clone)
        # cv2.waitKey(0)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    filename = "Images/Result/%d/output.jpg" % imageIndex
    cv2.imwrite(filename, image)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

print("time: ({} seconds)".format(time.time() - start_time))
