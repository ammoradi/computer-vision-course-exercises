What the program does?:
=============
* read static image from directory
* uses Opencv's Built-in Neat Image noise reduction [cv.fastNlMeansDenoising()](https://docs.opencv.org/3.4.3/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93)
* show denoised image

How to run?:
=============
* run app.py file by following command: `$ python3.6 app.py`

Limitation:
=============
only supports grayscale images.
to support colored image change `fastNlMeansDenoising()` function with `fastNlMeansDenoisingColored()` in app.js file [See Differences](https://docs.opencv.org/3.4.3/d1/d79/group__photo__denoise.html#ga03aa4189fc3e31dafd638d90de335617)

Refrences:
=============
* [OpenCv Denoising](https://docs.opencv.org/3.4.3/d5/d69/tutorial_py_non_local_means.html)
* [Neat Image](https://ni.neatvideo.com/overview/how-does-it-work)