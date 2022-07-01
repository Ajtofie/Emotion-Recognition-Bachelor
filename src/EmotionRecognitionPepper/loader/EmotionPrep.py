# coding=utf-8
# Import libraries
import os
import time

import cv2
import face_recognition
import numpy as np

labels_list = []

emotion_to_int = {"AF": 0, "AN": 1, "NE": 2, "SA": 3, "HA": 4}
int_to_emotion = {0: "AF", 1: "AN", 2: "NE", 3: "SA", 4: "HA"}
emotion_list = emotion_to_int.keys()


def get_label_from_filename(filename):
    """ Given a filename of the format 'NM.NE2.93.tiff', return the label 'NE'."""
    index = filename.find('.')
    return filename[index + 1:index + 3]


def prep_emotions_for_classifier_with_data_path(data_path):
    # Read the label and save emotions in numpy array
    print("reading the Emotion Labels...")
    for img in os.listdir(data_path):
        emotion = get_label_from_filename(img)
        emotion_int = emotion_to_int[emotion]  # convert to index
        labels_list.append(emotion)
    return labels_list
