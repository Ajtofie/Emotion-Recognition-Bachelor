# coding=utf-8
import sys
from src.EmotionRecognition.classifier import SVM
from src.EmotionRecognition.processor import ImageLoader, TrainTestSplit


resource_path = '..\\..\\resources'
path_2_emotions = resource_path + '\\2_Emotion_HaSa_DB'
path_3_emotions = resource_path + '\\3_Emotion_HaSaAn_DB'
path_4_emotions = resource_path + '\\4_Emotion_HaSaAnAf_DB'
path_5_emotions = resource_path + '\\5_Emotion_DB'
path_list = [path_2_emotions, path_3_emotions, path_4_emotions, path_5_emotions]
train_size = 0.8

for path in path_list:
    pre_processed, fully_processed, labels = ImageLoader.load_images_from_data_path(path)
    train_images, train_labels, test_images, test_labels = TrainTestSplit.split(pre_processed, labels, train_size)
    train_dataset, test_dataset = TrainTestSplit.reshape_data(train_images, test_images)
    SVM.classify(train_dataset, train_labels, test_dataset, test_labels, "Punktoperation")

    train_images, train_labels, test_images, test_labels = TrainTestSplit.split(fully_processed, labels, train_size)
    train_dataset, test_dataset = TrainTestSplit.reshape_data(train_images, test_images)
    SVM.classify(train_dataset, train_labels, test_dataset, test_labels, "Geometrische Operation")
sys.exit()





