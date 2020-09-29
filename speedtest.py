import cv2
import numpy as np

img = cv2.imread('asdf2.jpg', 0)
print(img)
cv2.imshow('before', img)
for x in range(100, 150):
    img[x, range(100, 150)] = 10
cv2.imshow('after', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
