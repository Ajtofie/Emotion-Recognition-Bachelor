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


# C: From the doc: "For larger values of C, a smaller margin will be accepted if the decision
# .  function is better at classifying all training points correctly. A lower C will encourage
#    a larger margin, therefore a simpler decision function, at the cost of training accuracy.
#     In other words ``C`` behaves as a regularization parameter in the SVM."
# OVR: One-versus-rest (alternative: ovo -- One-versus-one)
# Kernel (RBF): Radial Basis Functions
# Probability (False): Estimate the probability for class membership from scores
# class_weight (None): Give more weight to some classess
# coef0: Constant r in the kernel definition (see above)


def classify(train_images, train_labels, test_images, test_labels):
    classifier = svm.SVC(kernel='linear', gamma=0.001)
    print("classifier: ", classifier)

    nsamples, nx, ny = train_images.shape
    train_dataset = train_images.reshape((nsamples,nx*ny))

    nsamples, nx, ny = test_images.shape
    test_dataset = test_images.reshape((nsamples,nx*ny))

    # We learn the SVM model on the training data
    # classifier.fit(train_images, train_labels)
    #
    # # Now predict on the test data
    # predicted = classifier.predict(test_images)
    # expected = test_labels
    #
    # # print confusion matrix for visibility
    # src.EmotionRecognitionPepper.visualize.ConfusionMatrix.show_confusion_matrix(expected, predicted)
    #
    # print("Classification report for classifier %s:\n%s\n"
    #       % (classifier, metrics.classification_report(expected, predicted)))

    # do cross_validation to find the best hyperparameters
    if InputQuery.input_query("do cross validation"):
        print("Searching for best hyperparameters")
        cross_validation(train_dataset, train_labels)

    best_svm = svm.SVC(kernel='linear', C=8.68511373751352, gamma=1e-09)
    best_svm.fit(train_dataset, train_labels)

    predicted = best_svm.predict(test_dataset)
    expected = test_labels

    print("Classification report for classifier %s:\n%s\n"
          % (best_svm, metrics.classification_report(expected, predicted)))

    # print confusion matrix for visibility
    src.EmotionRecognitionPepper.visualize.ConfusionMatrix.show_confusion_matrix(expected, predicted)

    return best_svm


def cross_validation(train_images, train_labels):
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

    # testing = test_images[0:1]
    # # print(testing)
    # prediciton = best_SVM.predict(testing)
    # print(prediciton)
    # endregion

    # mlp_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 200), random_state=1, verbose=True)
    #
    # # alpha: L2 penalty (regularization term) parameter.
    # # beta_1, beta_2: parameters for first-order and second-order moments of Adam
    # # loss: cross-entropy loss.
    #
    # # We learn the SVM model on the training data
    # mlp_classifier.fit(train_images, train_labels)
    #
    # # Now predict on the test data
    # predicted = mlp_classifier.predict(test_images)
    # expected = test_labels
    #
    # print("Classification report for classifier %s:\n%s\n"
    #       % (mlp_classifier, metrics.classification_report(expected, predicted)))
