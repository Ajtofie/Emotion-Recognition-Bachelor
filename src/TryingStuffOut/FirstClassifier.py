# Tutorial from
# https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn
# Dataset used:
# https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB

# Load Dataset
data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Look at the data
# print label_names
# print labels
# print feature_names
# print features

# Split our data into Test and Training sets
train, test, train_labels, test_labels = tts(features, labels, test_size=0.33, random_state=42)

# Initialize our classifier
gnb = GaussianNB()

# Train our Classifier
model = gnb.fit(train,  train_labels)

# Make predictions
preds = gnb.predict(test)
# print preds

# Evaluate accuracy
print(accuracy_score(test_labels, preds))