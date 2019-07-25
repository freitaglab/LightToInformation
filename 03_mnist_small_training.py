################################################################################
# File:     03_mnist_small_training.py
# Author:   Michael Rinderle
# Email:    michael.rinderle@tum.de
# Created:  26.04.2019
#
# Revisions: ---
#
# Description: This script defines a 3 layer neural network, trains and
#              evaluates the network using the smaller 14x14 pixel MNIST dataset.
#              Accuracy of about 95% is reached.
#
################################################################################


import h5py
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# load MNIST dataset
with h5py.File("mnist_small.h5", "r") as file:
    train_images = np.array(file.get("train_images"))
    train_labels = np.array(file.get("train_labels"))
    test_images = np.array(file.get("test_images"))
    test_labels = np.array(file.get("test_labels"))

# normalize data
train_images = train_images / 256
test_images = test_images / 256


# define neural network layers
layer1 = keras.layers.Flatten(input_shape=train_images.shape[1:])       # flatten images (14x14 --> 1x196)
layer2 = keras.layers.Dense(32, activation=tf.nn.relu)                  # fully connected layer with 32 nodes      relu activation
layer3 = keras.layers.Dense(10, activation=tf.nn.softmax)               # dense layer with 10 nodes                softmax activation --> probabilities sum up to 1
# build model
model = keras.Sequential([layer1, layer2, layer3])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train the model
model.fit(train_images, train_labels, epochs=5)

# save the trained model to hdf5 file
model.save("trained_models/small_mnist_model.h5")


# test model performance
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test loss: {:.4f} - accuracy: {:.4f}".format(test_loss, test_accuracy))

predictions = model.predict(test_images)


# helper functions
def plot_image(image, true_label, prediction):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    predicted_label = np.argmax(prediction)

    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label, 100 * np.max(prediction), true_label), color=color)


def plot_prediction(true_label, prediction):
    thisplot = plt.bar(range(10), prediction, color="gray")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.ylim([0, 1])

    predicted_label = np.argmax(prediction)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


# plot first X items
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(test_images[i], test_labels[i], predictions[i])
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_prediction(test_labels[i], predictions[i])
plt.show()
