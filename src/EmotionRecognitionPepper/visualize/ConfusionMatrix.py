from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import src.EmotionRecognitionPepper.loader.EmotionPrep


def show_confusion_matrix(expected, predicted):
    cm = confusion_matrix(expected, predicted)
    plt.figure(figsize=(7, 5))
    sn.heatmap(cm, annot=True, xticklabels=src.EmotionRecognitionPepper.loader.EmotionPrep.emotion_list, yticklabels=src.EmotionRecognitionPepper.loader.EmotionPrep.emotion_list)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
