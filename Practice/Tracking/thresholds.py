import cv2 as cv
import numpy as np

img = cv.imread('images/UV_img_sample.jpg')
cv.imshow('UV', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

### SIMPLE ###
threshold, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY) # Whatever is greater than 150 will be 255, the rest is 0
cv.imshow('Simple threshold', thresh)

# threshold, thresh_inv = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
# cv.imshow('Simple threshold inverse', thresh_inv)

### ADAPTIVE ###
# adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)
# adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive threshold', adaptive)

cv.waitKey(0)