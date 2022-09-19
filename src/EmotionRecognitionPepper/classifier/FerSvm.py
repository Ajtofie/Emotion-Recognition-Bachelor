# coding=utf-8
# Import libraries
import os
import numpy as np
from sklearn import svm, metrics

import src.EmotionRecognitionPepper.visualize.ConfusionMatrix
import src.EmotionRecognitionPepper.processor.EmotionLoader
import src.EmotionRecognitionPepper.processor.ImageLoader
import src.EmotionRecognitionPepper.visualize.Camera as Camera
from src.EmotionRecognitionPepper.util import Timer, InputQuery
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


kernel_list = ['linear', 'rbf', 'sigmoid']


def classify(train_dataset, train_labels, test_dataset, test_labels):
    cross_validation(train_dataset, train_labels)  # TODO remove when done with it

    for kernel in kernel_list:
        classifier = svm.SVC(C=0.02, kernel=kernel, gamma=1e-09)
        classifier.fit(train_dataset, train_labels)

        predicted = classifier.predict(test_dataset)
        expected = test_labels

        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))

        # print confusion matrix for visibility
        src.EmotionRecognitionPepper.visualize.ConfusionMatrix.show_confusion_matrix(expected, predicted, "SVM " + kernel)

    return classifier


def cross_validation(train_images, train_labels):
    if InputQuery.input_query("do cross validation"):
        print("Searching for best hyperparameters")
        # Specify the ranges to be searched for hyper-parameters
        C_range = np.logspace(-2, 10, 5)
        gamma_range = np.logspace(-9, 3, 5)
        param_grid = dict(gamma=gamma_range, C=C_range)
        print("Cross validating")
        # Do cross validation
        cv = KFold(n_splits=5)  # default n_splits = 5
        grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=param_grid, cv=cv)
        grid.fit(train_images, train_labels)

        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
