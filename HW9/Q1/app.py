import cv2 as cv
import numpy as np


# cascade face detector
def face_detector(image):
    face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return 0, 0, 0, 0
    return faces[0]


if __name__ == '__main__':
    # read video
    video = cv.VideoCapture('input.avi')
    ret, frame = video.read()

    # get video frame's width and height
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    if ret:

        # create VideoWriter object.The output will be stored in 'output.avi' file.
        out = cv.VideoWriter(
            'output.avi',
            cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            10, (frame_width, frame_height)
        )

        # get 4 points of face rectangle
        x, y, w, h = face_detector(frame)

        # initialize Kalman filter parameters
        # 4 state(F), 2 measurement(H) and 0 control
        kalman = cv.KalmanFilter(4, 2, 0)
        # a rudimentary constant speed model:
        # x_t+1 = x_t + v_t
        kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                            [0., 1., 0., .1],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]])
        # measurement matrix (H)
        kalman.measurementMatrix = 1. * np.eye(2, 4)
        # gaussian distribution for process error (Q)
        kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
        # gaussian distribution for measurement error (R)
        kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
        kalman.errorCovPost = 1e-1 * np.eye(4, 4)
        kalman.statePost = np.array([x + w / 2, y + h / 2, 0, 0], dtype='float64')
        lw = w
        lh = h
        lx = x
        ly = y
        # do prediction, measurement and correction on each frame
        while True:
            ret, frame = video.read()
            if not ret:
                break
            x, y, w, h = face_detector(frame)

            lw = w if w != 0 else lw
            lh = h if h != 0 else lh
            ly = y if y != 0 else ly
            lx = x if x != 0 else lx

            # draw red rectangle of image detection
            img = cv.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (0, 0, 255), 3)
            # prediction
            predicted = kalman.predict()
            # measurement
            measurement = np.array([x + w / 2, y + h / 2], dtype='float64')
            # correction
            if x != 0 and w != 0 and y != 0 and h != 0:
                corrected = kalman.correct(measurement)
            else:
                corrected = predicted
            # draw green rectangle of kalman filter prediction's correction
            img2 = cv.rectangle(
                img, (int(corrected[0] - lw / 2), int(corrected[1] - lh / 2)),
                (int(corrected[0] + lw / 2), int(corrected[1] + lh / 2)), (0, 255, 0), 3
            )
            cv.imshow('frame', img)
            k = cv.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                out.write(img)

    # release the video
    out.release()

    # close all the frames
    cv.destroyAllWindows() 
