# coding=utf-8
# Import libraries
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import src.EmotionRecognitionPepper.processor.EmotionLoader
import src.EmotionRecognitionPepper.visualize.ConfusionMatrix


def classify(train_images, train_labels, test_images, test_labels):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_images, train_labels)
    # print knn.score(train_images, train_labels)

    y_pred = knn.predict(test_images)

    src.EmotionRecognitionPepper.visualize.ConfusionMatrix.show_confusion_matrix(test_labels, y_pred)

    print (classification_report(test_labels, y_pred))

    # List Hyperparameters that we want to tune.
    leaf_size = list(range(1, 100))
    n_neighbors = list(range(1, 100))
    p = [1, 2]
    # Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    # Create new KNN object
    knn_2 = KNeighborsClassifier()
    # Use GridSearch
    clf = GridSearchCV(knn_2, hyperparameters, cv=10)
    # Fit the model
    best_model = clf.fit(train_images, train_labels)
    # Print The value of best Hyperparameters
    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
