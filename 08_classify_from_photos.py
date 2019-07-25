################################################################################
# File:     08_classify_from_photos.py
# Author:   Michael Rinderle, Richard Freitag
# Email:    michael.rinderle@tum.de
# Created:  06.07.2019
#
# Revisions: ---
#
# Description: This script classifies MNIST digits from photos taken from a printout of MNIST digits
#              The accuracy is 127 out of 160 images using 180 as a value for the edge filter
#              and nearest neighbour interpolation
#
################################################################################


import serial
import time
import h5py
import numpy as np
import glob
import cv2
import os

# setup arduino serial communication
#arduino = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.050)
arduino = serial.Serial('COM22', 1000000, timeout=0.050)
print(arduino.name)

# reset arduino and whait for it to be ready
arduino.setDTR(False)
time.sleep(1)
arduino.setDTR(True)
while (True):
    line = arduino.readline()
    if "Arduino is ready" in line.decode():
        break

pred_images = 0
correct_images = 0
        
# high value for simple edge filter
highVal = 180
files = glob.glob("Raw_Photos/*.jpg")

for myFile in files:
    img_ = cv2.imread(myFile, cv2.IMREAD_ANYCOLOR)
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img_ = cv2.resize(gray,(14,14), interpolation = cv2.INTER_NEAREST)
    
    # bad pixels
    img_[9][0] = img_[10][0]
    
    captured_image = np.asarray(img_)
    high_values_flags = captured_image > highVal  # Where values are low
    captured_image[high_values_flags] = 255  # All low values set to 0
    
    newFileName = "Processed_Photos/" + os.path.basename(myFile)
    sep = ' '
    ref = os.path.basename(myFile).split(sep, 1)[0]
    print("Result should be %s" % ref)
    
    captured_image = cv2.bitwise_not(captured_image)
    img_resized = cv2.imwrite(filename=newFileName, img=captured_image)

    image = np.array(captured_image)
    image = np.reshape(image, (-1, 14*14))
    # send image to arduino
    arduino.write(b"S")
    for row in image:
        for num in row:
            arduino.write(bytes(str(num) + ", ", "utf-8"))
    arduino.write(b"E")

    # receive result from arduino
    while (True):
        if arduino.in_waiting:
            line = arduino.readline()
            print(line)

            if "R" in line.decode():
                data = arduino.readline()
                print("Prediction on Arduino: %d" % int(data))
                if (int(data) == int(ref)):
                    correct_images += 1
                
                pred_images += 1
                #print(int(data))
                break    

result = correct_images/pred_images
print("%d out of %d correct predicted" % (correct_images, pred_images))
print("Accuracy: {:.3f}".format(result))