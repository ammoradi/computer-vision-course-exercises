What the program does?:
=============
* read static image from directory
* uses Opencv's Built-in Histogram stretcher [cv.equalizeHist()](https://docs.opencv.org/3.4.3/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e)
* show stretched histogram
* save contrast stretched output image as a res.png

How to run?:
=============
* run app.py file by following command: `$ python3.6 app.py`

Limitation:
=============
* openCv has not any histogram stretcher. they implemented histogram equalizer which do stretching.

Refrences:
=============
* [OpenCv Histogram Equalization](https://docs.opencv.org/3.4.3/d5/daf/tutorial_py_histogram_equalization.html)
