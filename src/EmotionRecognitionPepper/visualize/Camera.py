# coding=utf-8
import cv2
import cv2 as cv
import face_recognition
import numpy as np
from sklearn import feature_extraction as extraction


def start_video_capture(classifier):
        if request_permission():
            faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

            capture = cv.VideoCapture(0 + cv.CAP_DSHOW)  # open_webcam()
            print("Access granted")
            while True:
                # Capture frame-by-frame
                ret, frame = capture.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                lol = np.array(frame, dtype='uint8')
                faces = faceCascade.detectMultiScale(lol, 1.1, 4)
                print 'faces=', faces

                for (x, y, w, h) in faces:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                font = cv.FONT_HERSHEY_SIMPLEX


                input_img_resize = cv2.resize(frame, (128, 128))  # ~ 0.00075 sekunden
                img_data = np.array(input_img_resize)
                img_data = img_data.astype('float32')
                img_data = img_data / 255  # Normalize between [0-1]
                x = img_data.reshape((len(img_data), -1)) # Flatten the images
                y = cv.resize(x, (128, 128))
                prediction = classifier.predict(y)
                cv.putText(frame,
                           prediction[0],
                           (50, 50),
                           font, 3,
                           (0, 0, 255),
                           2,
                           cv.LINE_4)
                # Display the resulting frame
                cv.imshow('Video Capture', frame)
                if cv.waitKey(1) == ord('q'):
                    capture.release()
                    cv.destroyAllWindows()
                    break


def start_video_capture_with_landmarks():
    if request_permission():
        faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        capture = cv.VideoCapture(0 + cv.CAP_DSHOW)  # open_webcam()
        print("Access granted")
        while True:
            # Capture frame-by-frame
            ret, frame = capture.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            lol = np.array(frame, dtype='uint8')
            faces = faceCascade.detectMultiScale(lol, 1.1, 4)
            print 'faces=', faces

            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            font = cv.FONT_HERSHEY_SIMPLEX

            input_img_landmarks = face_recognition.face_landmarks(frame)  # ~ 1.1 sekunden
            for a in input_img_landmarks:
                for key, values in a.items():
                    for (x, y) in values:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), 1)

            input_img_resize = cv2.resize(frame, (128, 128))  # ~ 0.00075 sekunden
            img_data = np.array(input_img_resize)
            img_data = img_data.astype('float32')
            img_data = img_data / 255  # Normalize between [0-1]
            x = img_data.reshape((len(img_data), -1)) # Flatten the images
            y = cv.resize(x, (128, 128))

            # Display the resulting frame
            cv.imshow('Video Capture', frame)
            if cv.waitKey(1) == ord('q'):
                capture.release()
                cv.destroyAllWindows()
                break


def request_permission():
    print("Starting video capture if webcam is existent")
    answer = raw_input("Webcam found. Do you want to open the webcam [y,n]?")
    if answer == "y" or answer == "Y": return True
    print("permission denied!")
    return False


def open_webcam():
    capture = cv.VideoCapture(1 + cv.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv.VideoCapture(0 + cv.CAP_DSHOW)
    if not capture.isOpened():
        raise IOError("Cannot open webcam!")
    return capture
