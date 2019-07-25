################################################################################
# File:     00_download_data.py
# Author:   Richard Freitag
# Email:    richard.freitag@uadm.uu.se
# Created:  26.07.2019
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

import requests
import zipfile
import os

url = 'https://zenodo.org/record/3351382/files/mnist_on_arduino.zip'
r = requests.get(url, allow_redirects=True)
open('mnist_on_arduino.zip', 'wb').write(r.content)

with zipfile.ZipFile('mnist_on_arduino.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
    
os.remove('mnist_on_arduino.zip')
print("Done...")