import cv2
from matplotlib import pyplot as plt
import glob

files = []
for file in glob.glob("images/*.jpg"):
    files.append(file)

for ix in files:
    img = cv2.imread(ix, cv2.IMREAD_COLOR)
    img_test_init = img.copy()
    img_test = cv2.cvtColor(img_test_init, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('data/haarcascade_mouth.xml')

    faces = face_cascade.detectMultiScale(img_test, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = img_test[y:y + h, x:x + w]
        roi_color = img_test_init[y:y + h, x:x + w]
        face_detect = cv2.rectangle(img_test_init, (x, y), (x + w, y + h), (255, 0, 255), 2)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        mouths = mouth_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_detect = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)
            plt.imshow(eye_detect)

        for (mx, my, mw, mh) in mouths:
            mouth_detect = cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 255), 2)
            plt.imshow(mouth_detect)

        cv2.imshow('face_detect', face_detect)
        cv2.waitKey(0)
