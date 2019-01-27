import cv2 as cv


def resize(img, height):
    (h, w) = img.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


if __name__ == '__main__':

    img_name = "yard"
    img_count = 9

    # initialize opencv's stitcher class
    stitcher = cv.createStitcher(False)

    pics = []
    for i in range(img_count):
        pics.append(cv.imread("./Images/" + img_name + "/" + img_name + "-0" + str(i) + ".png"))

    # get stitched image
    result = stitcher.stitch(tuple(pics))

    cv.imwrite("./Outputs/" + img_name + ".jpg", result[1])
    cv.imshow("result", resize(result[1], 500))
    cv.waitKey(0)
