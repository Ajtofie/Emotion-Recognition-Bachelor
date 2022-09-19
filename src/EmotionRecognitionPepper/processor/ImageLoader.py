# coding=utf-8
# Import libraries
import math
import sys
import os
import time
import src.EmotionRecognitionPepper.util.Timer
import cv2
from src.EmotionRecognitionPepper.visualize import Draw
from src.EmotionRecognitionPepper.processor import EmotionLoader, Detector
import numpy as np
import skimage
from skimage import io
from natsort import natsorted, ns

from src.EmotionRecognitionPepper.processor import ImageProcessor

file_name = 'C:\Users\Matej\IdeaProjects\pythonTest\src\EmotionRecognitionPepper\encodedImages.csv'


def load_images_from_data_path(data_path):
    image_list = []
    grayscale_images = []
    grayscale_processed_images = []
    landmarks_list = []
    landmark_proc_list = []
    emotion_list = []
    for img in os.listdir(data_path):

        if "jpg" not in img:
            print ("Not an image: " + img)
            continue

        '''Loading Images'''
        print("Loading image: " + img)
        image = io.imread(data_path + "/" + img)
        image_list.append(image)  # full images

        '''Process Image'''
        print("Processing image ...")
        # grayscale
        grayscale_img = ImageProcessor.pre_process(image)
        grayscale_images.append(grayscale_img)

        # processed
        processed_img, process_successful = ImageProcessor.process_image(image)
        if not process_successful:
            print("Image couldn't be processed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue
        grayscale_processed_images.append(processed_img)

        # landmarks
        landmarks = Detector.get_all_facial_landmarks_from(image, Detector.detect_face_from(image)[0])
        landmarks_list.append(landmarks)

        landmarks = Detector.get_all_facial_landmarks_from(processed_img, Detector.detect_face_from(processed_img)[0])
        landmark_proc_list.append(landmarks)
        # todo wenn image nicht processed wird, wird das ganze bild zurückgegeben sollte net so sein

        '''Loading Emotions'''
        print("Loading emotion ...")
        emotion_list.append(EmotionLoader.get_emotion(img))

        print("Finished loading image: " + img)

    return np.array(image_list), np.array(grayscale_images), np.array(grayscale_processed_images), np.array(landmarks_list), np.array(landmark_proc_list), np.array(emotion_list)


# save the encoded images as .csv file for faster loading times after first start
def data_saver(encoded_img_data_list):
    print("Saving data to " + file_name)
    np.savetxt('encodedImages.csv', encoded_img_data_list)


# Load images that are already saved in given file_name
def data_loader():
    print("Loading data from " + file_name)
    return np.loadtxt(file_name)
