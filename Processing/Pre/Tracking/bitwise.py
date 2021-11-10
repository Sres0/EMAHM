import cv2 as cv
import numpy as np

blank = np.zeros((400,400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1,)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1,)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

### BITWISE AND ###
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and) # Intersected regions

### OR ###
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwise_or) # Intersected regions

cv.waitKey(0)