import numpy as np
import cv2

print('Bruh')
img = cv2.imread('asdf2.jpg', 1)
cv2.imshow('bitch`', img)
cv2.waitKey(0)
img[100, 100] = [50,100, 200]
cv2.imshow('bitch2`', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
