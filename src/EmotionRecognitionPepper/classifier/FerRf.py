# coding=utf-8
# Import libraries

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import src.EmotionRecognitionPepper.visualize.ConfusionMatrix
from pprint import pprint
import numpy as np


def classify(train_images, train_labels, test_images, test_labels):
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    print("classifier: ", classifier)

    classifier.fit(train_images, train_labels)

    predicted = classifier.predict(test_images)
    expected = test_labels

    # print classifier.score(test_images, test_labels)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    src.EmotionRecognitionPepper.visualize.ConfusionMatrix.show_confusion_matrix(expected, predicted)

    rf = RandomForestClassifier(random_state=42)

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(rf.get_params())

    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(train_images, train_labels)

    evaluate(rf_random, test_images, test_labels)
    pprint(rf_random.best_params_)


def evaluate(model, test_images, test_labels):
    predictions = model.predict(test_images)
    print predictions
    print test_labels
    errors = abs(np.subtract(predictions, test_labels))
    print errors
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy
