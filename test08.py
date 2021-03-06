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

print(image_to_feature_vector(cv2.imread("asdf2.jpg"), (32, 32)))

imagePaths = list(paths.list_images("pics/"))
data = []
labels = []

print(imagePaths)

"""
model = Sequential()
model.add(Dense(768, input_dim=3072, bias_initializer="uniform", activation="relu"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(trainData, trainLabels, epochs=50, batch_size=128, verbose=1)

(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print("so.... loss={:.4f}, acc={:.4f}%".format(loss, accuracy*100))
"""
