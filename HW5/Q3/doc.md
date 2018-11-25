What the program does?:
=============
* read static image from directory
* uses Opencv's Built-in Threshold and Adaptive Threshold functions [cv.adaptiveThreshold()](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)
* show collection of thresholded image by six types: [`BINARY`,`BINARY_INV`,`TOZERO`,`TOZERO_INV`,
`Adaptive Mean Thresholding`, `Adaptive Gaussian Thresholding`]

How to run?:
=============
* run app.py file by following command: `$ python3.6 app.py`

Threshold vs. Adaptive Threshold:
=============
In simple thresholding, the threshold value is global, i.e., it is same for all the pixels in the image. Adaptive thresholding is the method where the threshold value is calculated for smaller regions and therefore, there will be different threshold values for different regions.

Adaptive Threshold Parameters:
=============
* src => Source 8-bit single-channel image.

* maxValue => Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.

* adaptiveMethod => Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C

* thresholdType => Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.

* C => Constant subtracted from the mean or weighted mean . Normally, it is positive but may be zero or negative as well.

Refrences:
=============
* [OpenCv Thresholding](https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html)
* [Opencv AdaptiveThreshold Parameters](https://stackoverflow.com/questions/28763419/adaptive-threshold-parameters-confusion)