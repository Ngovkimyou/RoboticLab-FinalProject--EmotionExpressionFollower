# This script checks and prints the versions of key libraries used in the test environment.
import tensorflow as tf
import cv2
import numpy as np
import sys

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
