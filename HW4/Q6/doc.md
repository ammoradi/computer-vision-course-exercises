What the program does?:
=============
* read static image from directory
* uses Opencv's Built-in HoughCircles function to detect circles [cv.HoughCircles()](https://docs.opencv.org/3.3.1/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d)
* show input image with black circles (detected circles)

How to run?:
=============
* run app.py file by following command: `$ python3.6 app.py`

Limitation:
=============
The `cv2.HoughCircles` function was able to detect only four of the circles instead of all five, leaving out the one in the center.
It’s due to the `minDist` parameter of `cv.HoughCircles()`. The center (x, y) coordinates for the large outer circle are identical to the center inner circle, thus the center inner circle is discarded.
Unfortunately, there is not a way around this problem unless we make minDist unreasonably small, and thus generating many “false” circle detections.

Refrences:
=============
* [Hough Circle Transform](https://docs.opencv.org/3.3.1/da/d53/tutorial_py_houghcircles.html)
* [Circle Detection](https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/)