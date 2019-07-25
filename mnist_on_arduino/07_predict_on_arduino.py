################################################################################
# File:     07_predict_on_arduino.py
# Author:   Michael Rinderle
# Email:    michael.rinderle@tum.de
# Created:  28.04.2019
#
# Revisions: ---
#
# Description: This script sends the small 14x14 pixel MNIST pictures 1 by 1
#              to the Arduino and gathers the prediction from the Arduino.
#              Predictions take ~ 45 ms/image
#
################################################################################


import serial
import time
import h5py
import numpy as np

# load MNIST dataset
with h5py.File("mnist_small.h5", "r") as file:
    test_images = np.array(file.get("test_images"))
    test_labels = np.array(file.get("test_labels"))

# flatten images
test_images = np.reshape(test_images, (-1, 14 * 14))
# print(test_images.shape)

# setup arduino serial communication
arduino = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.050)
print(arduino.name)

# reset arduino and whait for it to be ready
arduino.setDTR(False)
time.sleep(1)
arduino.setDTR(True)
while (True):
    line = arduino.readline()
    if "Arduino is ready" in line.decode():
        break

# define how many images should be sent and prdicted by arduino
img_count = 1000

# initialize array for predictions
arduino_predictions = np.zeros(img_count, dtype=np.int8)

start = time.time()

for idx in range(img_count):
    # send image to arduino
    arduino.write(b"S")
    for num in test_images[idx]:
        arduino.write(bytes(str(num) + ", ", "utf-8"))
    arduino.write(b"E")

    # receive result from arduino
    while (True):
        if arduino.in_waiting:
            line = arduino.readline()
            # print(line)

            if "RESULT" in line.decode():
                data = arduino.readline()
                break

    # store prediction in array
    arduino_predictions[idx] = int(data)

end = time.time()


arduino_wrongs = np.nonzero(arduino_predictions - test_labels[:img_count])[0].shape[0]

print("Wrong predictions from arduino:", arduino_wrongs)
print("Accuracy of arduino:  {:.2f} %".format((1 - arduino_wrongs / img_count) * 100))
print("Predictions took {:.2f} s --> {:.2f} ms/image".format(end - start, (end - start) / img_count * 1000))
