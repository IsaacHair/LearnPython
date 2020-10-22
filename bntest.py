from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os

def image_to_feature_vector(image, size=(32,32)):
    return cv2.resize(image, size).flatten()

data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)

    if i > 0 and i % 10 == 0:
        print("[info] processed {}/{}".format(i, len(imagePaths)))


