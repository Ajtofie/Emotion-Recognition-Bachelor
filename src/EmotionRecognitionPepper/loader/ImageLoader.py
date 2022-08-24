# coding=utf-8
# Import libraries
import math
import sys

import os
import time

import face_recognition

import src.EmotionRecognitionPepper.util.Timer

import cv2
import numpy as np

rows = 256
cols = 256

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_data_list = []
file_name = 'C:\Users\Matej\IdeaProjects\pythonTest\src\EmotionRecognitionPepper\encodedImages.csv'


def load_images_from_data_path(data_path):
    print("Loading .csv file if existent...")
    answer = raw_input(".csv file found. Do you want to load the file[y,n]?")
    if answer == "y" or answer == "Y":
        try:
            return data_loader()
        except IOError:
            print("No .csv file found")
    i = 0
    print("Reloading images...")
    # print("Reading and encoding the img files...")
    start_time = time.clock()
    for img in os.listdir(data_path):
        if "jpg" not in img: continue  # only processing .jpg files
        print("Loading image: " + img)
        # Reading, Resizing and Encoding image and finally saving the encoded image in numpy array
        try:
            input_img = cv2.imread(data_path + "/" + img)  # ~ 0.2-0.3 sekunden
            input_img_resize = cv2.resize(input_img, (rows, cols))  # ~ 0.00075 sekunden
            input_img_gray = cv2.cvtColor(input_img_resize, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(input_img_gray)
            # show_image(img, input_img_resize)
            img_data_list.append(clahe_image)

        except IndexError:
            print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
            quit()
    print ("Loading time: " + src.EmotionRecognitionPepper.util.Timer.print_passed_time(start_time))
    img_data = np.array(img_data_list)

    # data_saver(img_data)
    return img_data
    # return get_facial_landmark_coordinates(img_data)


def get_facial_landmark_coordinates(images):
    x_list = []
    y_list = []
    for img in images:
        img_landmarks = face_recognition.face_landmarks(img)
        for a in img_landmarks:
            for key, values in a.items():
                for (x, y) in values:
                    x_list.append(float(x))
                    y_list.append(float(y))
    x_mean = np.mean(x_list)  # find both coordinates of center of gravity
    y_mean = np.mean(x_list)
    x_central = [(x-x_mean) for x in x_list]  # Calculate the distance center <-> other points in both axes
    y_central = [(y-y_mean) for y in y_list]

    landmarks_vectorized = []
    for x, y, w, z in zip(x_central, y_central, x_list, y_list): # Store all landmarks in one list in the form of x1, y1, x2, y2 etc.
        landmarks_vectorized.append(w)
        landmarks_vectorized.append(z)
        mean_np = np.asarray((y_mean, x_mean))
        coor_np = np.asarray((z, w))
        dist = np.linalg.norm(coor_np-mean_np)
        landmarks_vectorized.append(dist)
        landmarks_vectorized.append((math.atan2(y, x) * 360) / (2 * math.pi))

    data = {'landmarks_vectorized': landmarks_vectorized}

    return data


def show_image(name, img):
    draw_landmarks()
    cv2.imshow(name, img)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()


def draw_landmarks():
    for a in input_img_landmarks:
        for key, values in a.items():
            for (x, y) in values:
                cv2.circle(input_img_grey, (x, y), 1, (0, 0, 255), 1)


# save the encoded images as .csv file for faster loading times after first start
def data_saver(encoded_img_data_list):
    print("Saving data to " + file_name)
    np.savetxt('encodedImages.csv', encoded_img_data_list)


# Load images that are already saved in given file_name
def data_loader():
    print("Loading data from " + file_name)
    return np.loadtxt(file_name)
