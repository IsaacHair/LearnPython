import cv2
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()

print(image_to_feature_vector(cv2.imread("asdf2.jpg"), (32, 32)))

model = Sequential()
model.add(Dense(768, input_dim=3072, init="uniform", activation="relu"))
model.add(tf.Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(tf.Dense(2))
model.add(tf.Activation("softmax"))
