import cv2, os, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()

print(image_to_feature_vector(cv2.imread("asdf2.jpg"), (32, 32)))

model = Sequential()
model.add(Dense(768, input_dim=3072, bias_initializer="uniform", activation="relu"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))
