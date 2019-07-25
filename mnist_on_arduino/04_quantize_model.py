################################################################################
# File:     04_quantize_model.py
# Author:   Michael Rinderle
# Email:    michael.rinderle@tum.de
# Created:  26.04.2019
#
# Revisions: ---
#
# Description: This script loads the trained model created by script
#              "03_mnist_small_training.py" and quantizes weights and biases.
#              Quantization is needed to store weights and biases efficiently
#              as 1-byte integers. This is necessary to save space and
#              computation power on microcontrollers.
#              The quantization scheme was adapted from the procect
#              "https://github.com/ARM-software/ML-KWS-for-MCU"
#
#              The quantized weights and biases are stored in a hdf5 file and
#              the file "network.h" is created to include in the Arduino code.
#
################################################################################


import h5py
import numpy as np

# load model weights and biases
with h5py.File("trained_models/small_mnist_model.h5", "r") as file:
    model = file.get("model_weights")
    layer1 = model.get("dense/dense")
    l1_weights = np.array(layer1.get("kernel:0"))
    l1_bias = np.array(layer1.get("bias:0"))
    layer2 = model.get("dense_1/dense_1")
    l2_weights = np.array(layer2.get("kernel:0"))
    l2_bias = np.array(layer2.get("bias:0"))

# print("Layer 1 weights:", l1_weights.shape)
# print("Layer 1 biases: ", l1_bias.shape)
# print("Layer 2 weights:", l2_weights.shape)
# print("Layer 2 biases: ", l2_bias.shape)

# find number of fractional bits to represent range
l1w_min, l1w_max = np.min(l1_weights), np.max(l1_weights)
l1b_min, l1b_max = np.min(l1_bias), np.max(l1_bias)
l2w_min, l2w_max = np.min(l2_weights), np.max(l2_weights)
l2b_min, l2b_max = np.min(l2_bias), np.max(l2_bias)

l1w_bits = 7 - int(np.ceil(np.log2(max(abs(l1w_min), abs(l1w_max)))))
l1b_bits = 7 - int(np.ceil(np.log2(max(abs(l1b_min), abs(l1b_max)))))
l2w_bits = 7 - int(np.ceil(np.log2(max(abs(l2w_min), abs(l2w_max)))))
l2b_bits = 7 - int(np.ceil(np.log2(max(abs(l2b_min), abs(l2b_max)))))

# print("Layer 1 weights:  min {:3.3f}, max {:3.3f}, bits {}".format(l1w_min, l1w_max, l1w_bits))
# print("Layer 1 biases:   min {:3.3f}, max {:3.3f}, bits {}".format(l1b_min, l1b_max, l1b_bits))
# print("Layer 2 weights:  min {:3.3f}, max {:3.3f}, bits {}".format(l2w_min, l2w_max, l2w_bits))
# print("Layer 2 biases:   min {:3.3f}, max {:3.3f}, bits {}".format(l2b_min, l2b_max, l2b_bits))

# calculate fixed point representation for weights and biases   (signed 8-bit integers)
fixed_l1w = np.array(np.round(l1_weights * 2 ** l1w_bits), dtype=np.int8)
fixed_l1b = np.array(np.round(l1_bias * 2 ** l1b_bits), dtype=np.int8)
fixed_l2w = np.array(np.round(l2_weights * 2 ** l2w_bits), dtype=np.int8)
fixed_l2b = np.array(np.round(l2_bias * 2 ** l2b_bits), dtype=np.int8)

# layer dimensions
img_size = fixed_l1w.shape[0]
l1_size = fixed_l1b.shape[0]
l2_size = fixed_l2b.shape[0]

# fractional bits of images
img_bits = 8

# export weights and biases for arduino
with open("network.h", "w") as file:
    file.write("#ifndef NETWORK_H\n#define NETWORK_H\n\n")
    file.write("const PROGMEM char l1_weights[{:d}][{:d}] = {{\n{{".format(l1_size, img_size))
    np.savetxt(file, fixed_l1w.T, fmt="%4d", delimiter=", ", newline="},\n{")
    file.seek(file.tell() - 3)
    file.write("};\n\n")

    file.write("const PROGMEM char l1_bias[{:d}] = {{".format(l1_size))
    np.savetxt(file, fixed_l1b, fmt="%4d", delimiter=", ", newline=", ")
    file.seek(file.tell() - 2)
    file.write("};\n\n")

    file.write("const PROGMEM char l2_weights[{:d}][{:d}] = {{\n{{".format(l2_size, l1_size))
    np.savetxt(file, fixed_l2w.T, fmt="%4d", delimiter=", ", newline="},\n{")
    file.seek(file.tell() - 3)
    file.write("};\n\n")

    file.write("const PROGMEM char l2_bias[{:d}] = {{".format(l2_size))
    np.savetxt(file, fixed_l2b, fmt="%4d", delimiter=", ", newline=", ")
    file.seek(file.tell() - 2)
    file.write("};\n\n")

    file.write("#define img_bits {}\n".format(img_bits))
    file.write("#define l1w_bits {}\n".format(l1w_bits))
    file.write("#define l1b_bits {}\n".format(l1b_bits))
    file.write("#define l2w_bits {}\n".format(l2w_bits))
    file.write("#define l2b_bits {}\n".format(l2b_bits))

    file.write("\n#define img_size {}\n".format(img_size))
    file.write("#define l1_size {}\n".format(l1_size))
    file.write("#define l2_size {}\n".format(l2_size))

    file.write("\n#endif // NETWORK_H")


# save fixed point model to hdf5 file
with h5py.File("trained_models/fixedpoint_mnist_model.h5", "w") as file:
    h5_l1w = file.create_dataset("layer1_weights", data=fixed_l1w)
    h5_l1b = file.create_dataset("layer1_bias", data=fixed_l1b)
    h5_l2w = file.create_dataset("layer2_weights", data=fixed_l2w)
    h5_l2b = file.create_dataset("layer2_bias", data=fixed_l2b)
    h5_l1w.attrs.create("bits", l1w_bits)
    h5_l1b.attrs.create("bits", l1b_bits)
    h5_l2w.attrs.create("bits", l2w_bits)
    h5_l2b.attrs.create("bits", l2b_bits)
