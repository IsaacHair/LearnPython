import cv2, os, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import tensorflow as tf
import numpy as np

def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()

img = cv2.imread("cat.0.jpg")
cv2.imshow("before", img)
cv2.waitKey(0)
cv2.imshow("intermediate", cv2.resize(img, (32, 32)))
cv2.waitKey(0)
cv2.imshow("after", image_to_feature_vector(img, (32, 32)))
cv2.waitKey(0)
cv2.destroyAllWindows()
