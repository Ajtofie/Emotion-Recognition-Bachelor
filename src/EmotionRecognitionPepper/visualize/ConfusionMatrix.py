from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import src.EmotionRecognitionPepper.processor.EmotionLoader

emotion_to_int = {"AF": 0, "AN": 1, "NE": 2, "SA": 3, "HA": 4}
emotion_list = emotion_to_int.keys()


def show_confusion_matrix(expected, predicted, title):
    cm = confusion_matrix(expected, predicted)
    plt.figure(figsize=(7, 5))
    sn.heatmap(cm, annot=True, xticklabels=emotion_list, yticklabels=emotion_list)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(title)
    plt.show()
