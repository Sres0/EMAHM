import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Histograms allow you to visualize the pixel distribution (say gray scale) density
img = cv.imread('Processing\Post\Bank\Imgs\Blank_0.png')
# cv.imshow('UV', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

### GRAYSCALE ###
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

circle = cv.circle(blank, (img.shape[1]//2 + 100,img.shape[0]//2 + 45), 100, 255, -1)
mask = cv.bitwise_and(gray, gray,mask=circle)
cv.imshow('Mask', mask)

# # gray_hist = cv.calcHist([gray], [0], None, [256], [0,256]) # Without mask
gray_hist = cv.calcHist([gray], [0], mask, [256], [0,256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins') # Bins represent the intervals of pixel intensities--0 being black and 255, white
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show() # Peak is the # of pixels that have an intensity of # of bins

### COLOR ###
mask = cv.circle(blank, (img.shape[1]//2 + 100,img.shape[0]//2 + 45), 100, 255, -1)
masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow('Mask', masked)

plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins') # Bins represent the intervals of pixel intensities--0 being black and 255, white
plt.ylabel('# of pixels')

colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    # hist = cv.calcHist([img], [i], mask, [256], [0,256])
    plt.plot(hist, col)
    plt.xlim([0,256])

plt.show() # Peak is the # of pixels that have an intensity of # of bins

cv.waitKey(0)