################################################################################
# File:     05_compare_models.py
# Author:   Michael Rinderle
# Email:    michael.rinderle@tum.de
# Created:  27.04.2019
#
# Revisions: ---
#
# Description: This script loads the quantized model created by
#              "04_quantize_model.py" and implements the prediction algorithm
#              like it would be on the Arduino microcontroller.
#              A comparison to the full precision tensorflow model is made.
#              The loss in accuracy due to quantization is in the range of 0.1%
#
################################################################################


import h5py
import numpy as np

import tensorflow as tf
from tensorflow import keras

# load fixed point model
with h5py.File("trained_models/fixedpoint_mnist_model.h5", "r") as file:
    layer1_weights = np.array(file.get("layer1_weights"))
    layer1_bias = np.array(file.get("layer1_bias"))
    layer2_weights = np.array(file.get("layer2_weights"))
    layer2_bias = np.array(file.get("layer2_bias"))
    l1w_bits = file["layer1_weights"].attrs.get("bits")
    l1b_bits = file["layer1_bias"].attrs.get("bits")
    l2w_bits = file["layer2_weights"].attrs.get("bits")
    l2b_bits = file["layer2_bias"].attrs.get("bits")

# layer dimensions
img_size = layer1_weights.shape[0]
l1_size = layer1_bias.shape[0]
l2_size = layer2_bias.shape[0]

# load MNIST dataset
with h5py.File("mnist_small.h5", "r") as file:
    test_images = np.array(file.get("test_images"))
    test_labels = np.array(file.get("test_labels"))

# flatten images
test_images = np.reshape(test_images, (-1, img_size))
print(test_images.shape)

# fractional bits of images
img_bits = 8


################################################################################
# compute neural network with fixed point data

# cast weights and images to 32-bit integers
weights1 = np.array(layer1_weights, dtype=np.int32)
weights2 = np.array(layer2_weights, dtype=np.int32)
images = np.array(test_images, dtype=np.int32)

# multiply layer 1 weight matrix and image vector(s)
layer1 = np.squeeze(np.matmul(weights1.T, np.expand_dims(images, -1)))
# shift bits to have same precision as layer 1 bias
layer1 = np.right_shift(layer1, img_bits + l1w_bits - l1b_bits)
# add layer 1 bias
layer1 += layer1_bias
# relu activation
layer1 *= layer1 > 0

# multiply layer 2 weight matrix and layer 1 vector(s)
layer2 = np.squeeze(np.matmul(weights2.T, np.expand_dims(layer1, -1)))
# shift bits to have same precision as layer 2 bias
layer2 = np.right_shift(layer2, l1b_bits + l2w_bits - l2b_bits)
# add layer 2 bias
layer2 += layer2_bias

# # softmax activation  (actually correct implementation but needs floating point exp function)
# layer2 = layer2 / 2 ** l2b_bits
# layer2 = (np.exp(layer2).T / np.sum(np.exp(layer2), 1)).T
# print(layer2.dtype)

# "hard"max (should give same results but easier to implement on Arduino)   probabilities don't add up to 1 obviously
fixedpoint_predictions = np.argmax(layer2, axis=1)


################################################################################
# use tensorflow model to compute predictions
small_model = keras.models.load_model("trained_models/small_mnist_model.h5")
# print(small_model.summary())

# reshape and normalize images
test_images = test_images.reshape(-1, 14, 14) / 256
# evaluate tensorflow model
loss, acc = small_model.evaluate(test_images, test_labels)
tensorflow_predictions = small_model.predict(test_images)
tensorflow_predictions = np.argmax(tensorflow_predictions, axis=1)


fixedpoint_wrongs = np.nonzero(fixedpoint_predictions - test_labels)[0].shape[0]
tensorflow_wrongs = np.nonzero(tensorflow_predictions - test_labels)[0].shape[0]

print("Wrong predictions with quantized model:", fixedpoint_wrongs)
print("Wrong predictions with tensorflow model:", tensorflow_wrongs)
print("Accuracy of quantized model:  {:.2f} %".format((1 - fixedpoint_wrongs / len(test_labels)) * 100))
print("Accuracy of tensorflow model: {:.2f} %".format((1 - tensorflow_wrongs / len(test_labels)) * 100))
