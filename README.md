# LightToInformation
MNIST image classification on Arduino hardware. The dataset and model to reproduce the accuracies reported in the paper will be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3351382.svg)](https://doi.org/10.5281/zenodo.3351382)
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
  * Requests 2.22
* Arduino 1.8.5

### Non-standard hardware
* Arduino Uno

## Installation guide
Install the required packages and clone the repository.
## Demo and instructions for use
To recreate the accuracies from the paper execute the scripts in the folder mnist_on_arduino in the following order:
1. Execute *00_download_data.py*
2. Execute *05_compare_models.py*
3. Upload *mnist_on_arduino.ino* to the Arduino Uno
4. Change the serial port in *07_predict_on_arduino.py* depending on your platform (e.g., */dev/ttyACM0* on Linux or *COM3* on Windows) and execute
5. Execute *08_classify_from_photos.py*

To create a completely new model, run the scripts in their respective order. Reupload *mnist_on_arduino.ino* with the newly created *network.h* to the Arduino Uno before executing *07_predict_on_arduino.py*.