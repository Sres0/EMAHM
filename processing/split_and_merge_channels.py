import cv2 as cv
import numpy as np

img = cv.imread('images/UV_img_sample.jpg')
# cv.imshow('UV', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

# cv.imshow('Blue', b)
cv.imshow('Green', g)
# cv.imshow('Red', r)
# cv.imshow('blue', blue)
cv.imshow('green', green)
# cv.imshow('red', red)

# merged = cv.merge([b,g,r])
# cv.imshow('Merged', merged)

cv.waitKey(0)