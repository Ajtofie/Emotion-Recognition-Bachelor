# coding=utf-8
# Import libraries
import os
import numpy as np
from sklearn import svm, metrics

import src.EmotionRecognitionPepper.visualize.ConfusionMatrix
import src.EmotionRecognitionPepper.loader.EmotionPrep
import src.EmotionRecognitionPepper.loader.ImageLoader
import src.EmotionRecognitionPepper.visualize.Camera as Camera
from src.EmotionRecognitionPepper.util import Timer, InputQuery
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def classify(train_images, train_labels, test_images, test_labels):
    # TODO redo that stuff here
    nsamples, nx, ny = train_images.shape
    train_dataset = train_images.reshape((nsamples,nx*ny))

    nsamples, nx, ny = test_images.shape
    test_dataset = test_images.reshape((nsamples,nx*ny))

    cross_validation(train_dataset, train_labels)  # TODO remove when done with it

    classifier = svm.SVC(kernel='linear', C=8.68511373751352, gamma=1e-09) # TODO go through all different kernels
    classifier.fit(train_dataset, train_labels)

    predicted = classifier.predict(test_dataset)
    expected = test_labels

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    # print confusion matrix for visibility
    src.EmotionRecognitionPepper.visualize.ConfusionMatrix.show_confusion_matrix(expected, predicted)

    return classifier


def cross_validation(train_images, train_labels):
    if InputQuery.input_query("do cross validation"):
        print("Searching for best hyperparameters")
        cross_validation(train_images, train_labels)
    # Specify the ranges to be searched for hyper-parameters
    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-9, 3, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)

    # Do cross validation
    cv = KFold(n_splits=5)  # default n_splits = 5
    grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=param_grid, cv=cv)
    grid.fit(train_images, train_labels)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
