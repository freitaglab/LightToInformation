################################################################################
# File:     01_mnist_preprocessing.py
# Author:   Michael Rinderle
# Email:    michael.rinderle@tum.de
# Created:  26.04.2019
#
# Revisions: ---
#
# Description: This script downloads the MNIST dataset of hand written digits
#              and saves them in a hdf5 file.
#              It also reduces the image size from 28x28 pixels to 14x14 pixels
#              by a max pooling operation of 2x2 windows with stride 2.
#              The smaller images are saved in a separate hdf5 file.
#
################################################################################


import h5py
import numpy as np

import tensorflow as tf
from tensorflow import keras

# download MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# save dataset to hdf5 file
with h5py.File("mnist.h5", "w") as file:
    file.create_dataset("train_images", data=train_images)
    file.create_dataset("train_labels", data=train_labels)
    file.create_dataset("test_images", data=test_images)
    file.create_dataset("test_labels", data=test_labels)


# make images smaller  (28x28 --> 14x14)
pic_size = 14

train_small = np.zeros((len(train_images), pic_size, pic_size), dtype=np.uint8)
test_small = np.zeros((len(test_images), pic_size, pic_size), dtype=np.uint8)

for pic in range(len(train_images)):
    for i in range(0, pic_size):
        for j in range(0, pic_size):
            train_small[pic, i, j] = np.max(train_images[pic, 2 * i:2 * i + 2, 2 * j:2 * j + 2])

for pic in range(len(test_images)):
    for i in range(0, pic_size):
        for j in range(0, pic_size):
            test_small[pic, i, j] = np.max(test_images[pic, 2 * i:2 * i + 2, 2 * j:2 * j + 2])


# save smaller dataset to hdf5 file
with h5py.File("mnist_small.h5", "w") as file:
    file.create_dataset("train_images", data=train_small)
    file.create_dataset("train_labels", data=train_labels)
    file.create_dataset("test_images", data=test_small)
    file.create_dataset("test_labels", data=test_labels)
