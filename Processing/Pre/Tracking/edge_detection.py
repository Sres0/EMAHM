import cv2 as cv
import numpy as np

img = cv.imread('images/UV_img_sample.jpg')
cv.imshow('UV', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('B&W', gray)

### LAPLACIAN ###
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

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

ret, thresh = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # The most important
print(f'{len(contours)} contours found')
blank = np.zeros(img.shape, dtype='uint8')
cv.drawContours(blank, contours, -1, (255,255,255), 1)
cv.imshow('Blank', blank)

cv.waitKey(0)