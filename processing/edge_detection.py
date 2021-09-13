import cv2 as cv
import numpy as np

img = cv.imread('images/UV_img_sample.jpg')
cv.imshow('UV', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('B&W', gray)

### LAPLACIAN ###
# lap = cv.Laplacian(gray, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)

### SOBEL ###
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined', combined)

### CANNY ###
canny = cv.Canny(gray, 40, 90)
cv.imshow('Canny', canny)

cv.waitKey(0)