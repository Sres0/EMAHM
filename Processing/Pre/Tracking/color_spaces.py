import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Processing/Post/Bank/Imgs/hand_test_0.png')
w = int(1280/3)
h = int(720/3)
img = cv.resize(img, (w,h))
cv.imshow('Original', img) # Normal format is BGR

### GRAY ###
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('B&W', gray)

### HSV ###
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

### LAB ###
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

### RGB ###
# plt.imshow(img)
# plt.show()

rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

### BGR ### You can't convert from, say, rgb to hsv, but you can from rgb to bgr to hsv
rgb_bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
cv.imshow('RGB to BGR', rgb_bgr)
bgr_hsv = cv.cvtColor(rgb_bgr, cv.COLOR_BGR2HSV)
cv.imshow('BGR to HSV', bgr_hsv)

cv.waitKey(0)