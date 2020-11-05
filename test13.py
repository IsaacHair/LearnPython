import cv2, os, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Reshape

def image_to_feature_vector(image, size=(64, 64)):
    return cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2GRAY)

imagePaths = list(paths.list_images("lotspicsseparate/"))
data = []
labels = []
olddata = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    features = image_to_feature_vector(image)
    features = Reshape((features, 5, 8, 1))
    olddata.append(features)
    labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)

data = np.array(olddata) / 255.0
labels = np_utils.to_categorical(labels, 2)

(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)

model = Sequential()
model.add(Conv2D(64, (3,3), activation="relu", input_shape=(64,64)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(trainData, trainLabels, epochs=50, batch_size=128, verbose=1, validation_split=0.3, shuffle=True)

(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print("Results: loss={:.4f}, acc={:.4f}%".format(loss, accuracy*100))


print("Enter test file path: ")
testpath = input()
testimg = cv2.imread(testpath)
features = image_to_feature_vector(testimg)
features = np.array([features]) / 255.0

CLASSES = ["cat", "dog"]
probs = model.predict(features)[0]
prediction = probs.argmax(axis=0)
label = "{}: {:.2f}%".format(CLASSES[prediction], probs[prediction]*100)
cv2.putText(testimg, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
cv2.imshow("IMAGE", testimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
