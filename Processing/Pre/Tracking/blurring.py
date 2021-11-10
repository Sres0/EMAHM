import cv2 as cv

img = cv.imread('images/UV_img_sample.jpg')
# cv.imshow('UV', img)

### AVERAGING: averages the surrounding pixels of the center of the kernel ###
# average = cv.blur(img, (3,3)) # kernel is a matrix with any # of rows and columns, each with own value
average = cv.blur(img, (7,7)) # more blur
cv.imshow('Average blur', average)

### GAUSSIAN: usually less than average but more natural ###
# gauss = cv.GaussianBlur(img, (3,3), 0) # SigmaX = standard deviation in X
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow('Gaussian blur', gauss)

### MEDIAN: median of the surrounding pixels of the center of the kernel. Can be more effective at reducing noise ###
# median = cv.medianBlur(img, 3) #SigmaX = standard deviation in X
median = cv.medianBlur(img, 7)
cv.imshow('Median blur', median)

### BILATERAL: blurs but retains the edges ###
bilateral = cv.bilateralFilter(img, 10, 30, 30) # Diameter instead of kernel, SigmaColor more colors considered in the neighbors, sigmaSpace > pixels further out from central will influence the center
cv.imshow('Bilateral blur', bilateral)

cv.waitKey(0)
