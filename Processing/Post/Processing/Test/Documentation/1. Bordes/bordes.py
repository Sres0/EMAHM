import cv2 as cv
import numpy as np

i = 76
path = f'Processing/Post/Bank/Imgs/hand_test_{i}.png'
img_description = 'grayscale'

img = cv.imread(path)
w = int(1280/3)
h = int(720/3)
img = cv.resize(img, (w,h))
cv.imshow('Original', img)

### Escala gris ###
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Escala de gris', gray)

### Laplacian ###
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap)) # Considera valores positivos y negativos--no elimina información
cv.imshow('Laplacian', lap)

### Sobel ###
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0) # CV 64 por defecto
cv.imshow('sobel X', sobelx)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
cv.imshow('sobel Y', sobely)
combined = cv.bitwise_or(sobelx, sobely)
cv.imshow('sobel combinados', combined)
cv.imwrite(f'Processing/Post/Processing/Test/Documentation/1. Bordes/aooooooo_{i}.png', combined)

### Canny ###
canny = cv.Canny(gray, 40, 90)
cv.imshow('canny', canny)

### Threshold ###
_, thresh = cv.threshold(gray, 105, 255, cv.THRESH_BINARY) # Thresh binary por defecto
cv.imshow('Threshold', thresh)

cv.waitKey(0)