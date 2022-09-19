# coding=utf-8
# Import libraries

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import src.EmotionRecognitionPepper.visualize.ConfusionMatrix
from pprint import pprint
import numpy as np

max_features_list = {"sqrt", "log2", "auto"}


def classify(train_images, train_labels, test_images, test_labels):
    for kernel in max_features_list:
        classifier = RandomForestClassifier(bootstrap=False, max_depth=80, max_features=kernel, min_samples_leaf=1,
                                            min_samples_split=5, n_estimators=1400)
        print("classifier: ", classifier)

        classifier.fit(train_images, train_labels)

        predicted = classifier.predict(test_images)
        expected = test_labels

        # print classifier.score(test_images, test_labels)
        # evaluate(classifier, test_images, test_labels)
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))

        src.EmotionRecognitionPepper.visualize.ConfusionMatrix.show_confusion_matrix(expected, predicted, "RF" + kernel)
    return classifier


def evaluate(model, test_images, test_labels): # TODO refactor since it doesnt work
    errors = 0
    predictions = model.predict(test_images)
    print predictions
    print test_labels
    for i in predictions:
        if predictions[i] != test_labels[i]:
            print(predictions[i] + " != " + test_labels[i])
            errors = errors + 1
    print errors
    mape = 100 * np.mean(errors / 5)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy
