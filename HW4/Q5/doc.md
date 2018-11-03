What the programs does?:
=============
* both : read static image from directory
* lsd.py : uses Opencv's Built-in Line Segment Detector to detect lines [cv.fastNlMeansDenoising()](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#createlinesegmentdetector)
* hough.py : uses Opencv's Built-in Hough Line Detector to detect lines [cv.HoughLinesP()](https://docs.opencv.org/3.4.0/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb)
* show lines and rectangles

How to run?:
=============
* run lsd.py | hough.py file by following command: `$ python3.6 lsd.py`

Refrences:
=============
* [Hough Line Detection](https://docs.opencv.org/3.4.0/d9/db0/tutorial_hough_lines.html)
* [Lane Detection](https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0)