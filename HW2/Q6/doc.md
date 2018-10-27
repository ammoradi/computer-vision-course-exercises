What the program does?:
=============
* read two static images from directory
* does **Homogeneous blur** using [cv.blur()](https://docs.opencv.org/3.4.3/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)
* does **Guassian blur** using [cv.GaussianBlur()](https://docs.opencv.org/3.4.3/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)
* does **Median blur** using [cv.medianBlur()](https://docs.opencv.org/3.4.3/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)
* show six output images => 3 algorithms of two image
* save six output images

How to run?:
=============
* run app.py file by following command: `$ python3.6 app.py`

Refrences:
=============
* [OpenCv Image Smoothing](https://docs.opencv.org/3.4.3/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html)

Notation:
=============
* this program uses *Homogeneous blur* and *Guassian blur* as **Smoother**s