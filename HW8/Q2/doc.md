What the program does?:
=============
* read HaarCascade faces, eyes and lips Xmls from `/data` directory using openCv's built-in method [cv2.CascadeClassifier](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html). 
* read Face containing images from `/images` directory. 
* uses built-in openCv [cv2.detectMultiScale()](https://docs.opencv.org/3.0-beta/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale) to detect faces, eyes and lips.
* show all face detection iterations.

cv2.detectMultiScale() Parameters
=============
* **Image**: The first input is the grayscale image.
* **scaleFactor**: This function compensates a false perception in size that occurs when one face appears to be bigger than the other simply because it is closer to the camera. this parameter starts from 1 nad bigger numbers skips more details. we used 1.1 to see more details.
* **minNeighbors**: Detection algorithm that uses a moving window to detect objects, it does so by defining how many objects are found near the current one before it can declare the face found. the bigger number of this parameter detects more details. we use 5, the bigger number detects negative details for our useCase.

How to run?:
=============
* run app.py file by following command: `$ python3.6 app.py`
* after first image showed, press any key to see other iterations until all images be shown.

References:
=============
* [Python OpenCv FaceDetection](https://medium.com/analytics-vidhya/how-to-build-a-face-detection-model-in-python-8dc9cecadfe9).