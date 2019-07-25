# LightToInformation
## System requirements
The software was tested on the following operating systems/packages:
Operating system and dependencies including tested versions
* Windows/Linux/Mac
* Python 3.7.2
  * Tensorflow 1.1
  * H5py 2.9.0
  * NumPy 1.16.1
  * Matplotlib 3.0.2
  * OpenCV 4.1.0
* Arduino 1.8.5

### Non-standard hardware
* Arduino Uno

## Installation guide
Install the required packages and clone the repository.
## Demo and instructions for use
The scripts are numbered and should be run in their respective order. Upload the sketch *mnist_on_arduino.ino* to the Arduino Uno before running *07_predict_on_arduino.py*. Change the serial port depending on your platform (e.g., */dev/ttyACM0* on Linux or *COM3* on Windows).