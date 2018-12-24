What the program does?:
=============
* read template image ( the image will be found in other images ) from `Images/Template` directory.
* read flag images ( images for matching templates ) from `Images/Flags` directory.
* uses built-in openCv [cv2.matchTemplate()](https://docs.opencv.org/3.4/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be) Function to match images with template.
* uses built-in openCv [cv2.minMaxLoc()](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gab473bf2eb6d14ff97e89b355dac20707) Function to find matched points (finds the global minimum and maximum in an array).
* save all iterations and output images in `Images/Result` directory

How to run?:
=============
* run app.py file by following command: `$ python3.6 app.py`

Problem
=============
flag images have different sizes and it causes problem for template matching.

Solution
=============
first make template smaller (because for some cases, template is bigger than images).
then for each flag images, iterate algorithm for different scales of flag images and then find the best match (for each iteration, flag image will be smaller).

Method Comparison
=============
* **TM_CCOEFF_NORMED**

    time: 0.9316685199737549 seconds
    
    precision: GOOD
    
* **TM_CCORR_NORMED**

    time: 0.9328722953796387 seconds
    
    precision: GOOD
    
* **TM_SQDIFF_NORMED**

    time: 0.9070234298706055 seconds
    
    precision: MEDIUM

Refrences:
=============
* [Template Matching](https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html)
* [Multi-scale Template Matching](https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/)