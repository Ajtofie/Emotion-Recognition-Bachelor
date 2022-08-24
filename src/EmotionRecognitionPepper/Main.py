# coding=utf-8

# Import libraries
import numpy as np
from src.EmotionRecognitionPepper.classifier import FerSvm, FerRf
from src.EmotionRecognitionPepper.loader import ImageLoader, EmotionPrep
from src.EmotionRecognitionPepper.visualize import Camera

# TODO change datapath to sth
data_path = 'C:\\Users\\Matej\\IdeaProjects\\pythonTest\\resources\\dataset'
img_data = ImageLoader.load_images_from_data_path(data_path)
num_images = img_data.shape[0]  # TODO wtf ist des eig?
# TODO process image here
# TODO facial landmark processor
# TODO maybe encoding processor?

# Read the label and save emotions in numpy array
labels = np.array(EmotionPrep.prep_emotions_for_classifier_with_data_path(data_path))

# TODO split with testsplitdata stuff
# Split the data into train and test set
print("Splitting data into train andF test set...")
train_size = int(num_images * 0.8)  # reserve 80% for training, 20% for testing
train_images = img_data[0:train_size]
train_labels = labels[0:train_size]
test_images = img_data[train_size:]
test_labels = labels[train_size:]

print("... finished with the input and labels as follows: ")
print("-- train_images.shape: ", train_images.shape)
print("-- train_labels.shape: ", train_labels.shape)
print("-- test_images.shape: ", test_images.shape)
print("-- test_labels.shape: ", test_labels.shape)
print("-- The number of images: ", num_images)

svm_classifier = FerSvm.classify(train_images, train_labels, test_images, test_labels)
# FerRf.classify(train_images, train_labels, test_images, test_labels)
# FerKnn.classify(train_images, train_labels, test_images, test_labels)

# Emotion Recognition via Video (Webcam)
# Camera.start_video_capture(svm_classifier)

