# coding=utf-8
# Import libraries
import sys

import os
import time

import src.EmotionRecognitionPepper.util.Timer

import cv2
import numpy as np

rows = 256
cols = 256

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

    print("Reloading images...")
    # print("Reading and encoding the img files...")
    start_time = time.clock()
    for img in os.listdir(data_path):
        if "jpg" not in img: continue  # only processing .jpg files
        # Reading, Resizing and Encoding image and finally saving the encoded image in numpy array
        try:
            input_img = cv2.imread(data_path + "/" + img)  # ~ 0.2-0.3 sekunden
            input_img_resize = cv2.resize(input_img, (rows, cols))  # ~ 0.00075 sekunden
            # input_img_encoded = face_recognition.face_encodings(input_img, num_jitters=10)[0]  # ~ 1.1 sekunden

            img_data_list.append(input_img_resize)
        except IndexError:
            print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
            quit()
    print ("Loading time: " + src.EmotionRecognitionPepper.util.Timer.print_passed_time(start_time))
    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data = img_data / 255  # Normalize between [0-1]
    img_data = img_data.reshape((len(img_data), -1))  # Flatten the images

    # data_saver(img_data)

    return img_data


# save the encoded images as .csv file for faster loading times after first start
def data_saver(encoded_img_data_list):
    print("Saving data to " + file_name)
    np.savetxt('encodedImages.csv', encoded_img_data_list)


# Load images that are already saved in given file_name
def data_loader():
    print("Loading data from " + file_name)
    return np.loadtxt(file_name)
