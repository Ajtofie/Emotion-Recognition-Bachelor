import numpy as np
from sklearn.model_selection import train_test_split


def split(image_data, label_data, train_size):
    print("Splitting data into train and test set...")
    train_images, test_images, train_labels, test_labels = \
        train_test_split(image_data, label_data, train_size=train_size, random_state=42)
    # random_state=42 gives always same train_test_split

    print("... finished with the input and labels as follows: ")
    print("-- train_images.shape: ", train_images.shape)
    print("-- train_labels.shape: ", train_labels.shape)
    print("-- test_images.shape: ", test_images.shape)
    print("-- test_labels.shape: ", test_labels.shape)
    print("-- The number of images: ", image_data.shape[0])

    return train_images, train_labels, test_images, test_labels


def reshape_data(train, test):
    n_samples, nx, ny = train.shape[:3]
    train = train.reshape((n_samples, nx * ny))

    n_samples, nx, ny = test.shape[:3]
    test = test.reshape((n_samples, nx * ny))
    return train, test