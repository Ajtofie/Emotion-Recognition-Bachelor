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


height = 256
width = 256
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_data_list = []
file_name = 'C:\Users\Matej\IdeaProjects\pythonTest\src\EmotionRecognitionPepper\encodedImages.csv'


def load_images_from_data_path(data_path):
    for img in os.listdir(data_path):

        if "jpg" not in img:
            print ("Not an image: " + img)
            continue

        print("Loading and processing image: " + img)
        image = cv2.imread(data_path + "/" + img)
        image_resized = cv2.resize(image, (height, width))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        image_clahe = clahe.apply(image_gray) # TODO was ist das?

        img_data_list.append(image_clahe)


    return np.array(img_data_list)


# save the encoded images as .csv file for faster loading times after first start
def data_saver(encoded_img_data_list):
    print("Saving data to " + file_name)
    np.savetxt('encodedImages.csv', encoded_img_data_list)


# Load images that are already saved in given file_name
def data_loader():
    print("Loading data from " + file_name)
    return np.loadtxt(file_name)
