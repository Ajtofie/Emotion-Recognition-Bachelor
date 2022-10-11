# coding=utf-8
from sklearn import svm, metrics
from src.EmotionRecognition.visualize import ConfusionMatrix


def classify(train_dataset, train_labels, test_dataset, test_labels, title):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(train_dataset, train_labels)

    predicted = classifier.predict(test_dataset)
    expected = test_labels

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    ConfusionMatrix.show_confusion_matrix(expected, predicted, "SVM - " + title)
