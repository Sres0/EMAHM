import cv2 as cv
import numpy as np

### NORMAL IMAGE ###
# img = cv.imread('UV_img_sample.jpg')
# cv.imshow('Hand', img)

### COLOR ###
blank = np.zeros((500,500,3), dtype='uint8') #image code. 3 is color channels
# cv.imshow('Blank', blank)
# blank[:] = 0,255,0 #All green
# blank[200:300, 100:200] = 0,0,255 #

### RECTANGLE ###
# cv.rectangle(blank, (0,0), (250,500), (255,0,0), thickness=2)
# cv.rectangle(blank, (0,0), (250,500), (255,0,0), cv.FILLED) #Instead of just border; same as -1
cv.rectangle(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//4), (255,0,0), cv.FILLED) #relative to size
cv.imshow('Color', blank)

### CIRCLE ###
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=2)
cv.imshow('Color', blank)

### LINE ###
cv.line(blank, (0,0), (250,250), (255,255,255), thickness=2)
cv.imshow('Color', blank)

### TEXT ###
cv.putText(blank, 'Image', (300,450), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), thickness=2)
cv.imshow('Color', blank)

cv.waitKey(0)