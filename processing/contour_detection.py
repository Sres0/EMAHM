import cv2 as cv
import numpy as np

img = cv.imread('images/UV_img_sample.jpg')
# cv.imshow('UV', img)

blank = np.zeros(img.shape, dtype='uint8')
# cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

# canny = cv.Canny(img, 50, 150)
canny = cv.Canny(blur, 50, 150)
cv.imshow('Canny', canny)

ret, thresh = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) # All the contours
# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # The most important
# contours, hierarchies = cv.findContours(blur, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # The most important
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # The most important
print(f'{len(contours)} contours found')

cv.drawContours(blank, contours, -1, (255,255,255), 1)
cv.imshow('Blank', blank)

cv.waitKey(0)