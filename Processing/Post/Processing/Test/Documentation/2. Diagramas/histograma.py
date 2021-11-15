import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 76
j = i + 1
k = j + 1

def path(i):
    return f'Processing/Post/Bank/Imgs/hand_test_{i}.png'

def createImg(path):
    img = cv.imread(path)
    w = int(1280/3)
    h = int(720/3)
    return cv.resize(img, (w,h))

path1 = path(i)
path2 = path(j)
path3 = path(k)

img1 = createImg(path1)
img2 = createImg(path2)
img3 = createImg(path3)

cv.imshow('Original', img1)

### GRAYSCALE ###
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imshow(f'Escala de gris {i}', gray)

gray_hist = cv.calcHist([gray], [0], None, [256], [0,256])

plt.figure()
plt.title(f'Histograma escala de gris {i}')
plt.xlabel('Intensidad de pixel (0 - 255)')
plt.ylabel('# de pixeles')
plt.grid()
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

cv.waitKey(0)