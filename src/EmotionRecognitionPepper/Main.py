# coding=utf-8

# Import libraries
import sys

import numpy as np
from src.EmotionRecognitionPepper.classifier import FerSvm, FerRf, FerKnn
from src.EmotionRecognitionPepper.processor import ImageLoader, TrainTestSplit, ImageProcessor
from src.EmotionRecognitionPepper.visualize import Camera, Draw

data_path = '..\\..\\resources'
path_dataset = data_path + "\\dataset"
path_averted_and_straight = data_path + '\\dataset_averted_and_straight'
path_averted = data_path + '\\dataset_averted'
path_straight = data_path + '\\dataset_straight'
path_landmarks = data_path + '\\dataset_for_landmarks'
train_size = 0.8

# img_data_AS, labels_AS = ImageLoader.load_images_from_data_path(path_averted_and_straight)
# img_data_A, labels_A = ImageLoader.load_images_from_data_path(path_averted)
img_data, gray_data, gray_proc_data, landmark_data, landmark_proc_data, labels = ImageLoader.load_images_from_data_path(path_landmarks)
# TODO process image her
# TODO maybe encoding processor?

# Split the data into train and test set

train_images, train_labels, test_images, test_labels = TrainTestSplit.split(img_data, labels, train_size)

n_samples, nx, ny = train_images.shape
train_dataset = train_images.reshape((n_samples, nx * ny))

n_samples, nx, ny = test_images.shape
test_dataset = test_images.reshape((n_samples, nx * ny))

svm_classifier = FerSvm.classify(train_dataset, train_labels, test_dataset, test_labels)
# rf = FerRf.classify(train_images, train_labels, test_images, test_labels)
# FerKnn.classify(train_dataset, train_labels, test_dataset, test_labels)

# Emotion Recognition via Video (Webcam)
Camera.start_video_capture(1)
# Camera.start_video_capture(rf)
sys.exit()
