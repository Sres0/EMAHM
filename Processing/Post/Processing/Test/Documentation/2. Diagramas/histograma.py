import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 64
j = 76
k = j + 1

def path(i):
    return f'Processing/Post/Bank/Imgs/hand_test_{i}.png'

def createImg(path, i):
    img = cv.imread(path)
    w = int(1280/3)
    h = int(720/3)
    resized = cv.resize(img, (w,h))
    cv.imshow(f'Original {i}', resized)
    cv.imwrite(f'Processing/Post/Processing/Test/Documentation/2. Diagramas/original_{i}.png', resized)
    return resized

path1 = path(i)
path2 = path(j)
path3 = path(k)

img1 = createImg(path1, i)
img2 = createImg(path2, j)
img3 = createImg(path3, k)

### GRAYSCALE ###

def gray(i, img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow(f'Escala de gris {i}', gray)
    cv.imwrite(f'Processing/Post/Processing/Test/Documentation/2. Diagramas/gris_{i}.png', gray)
    return gray

def histogram(title, img):
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    plt.figure()
    plt.title(title)
    plt.xlabel('Intensidad de pixel (0 - 255)')
    plt.ylabel('Número de pixeles')
    plt.grid()
    plt.plot(hist)
    plt.xlim([0,256])

gray1 = gray(i, img1)
gray2 = gray(j, img2)
gray3 = gray(k, img3)

hist1 = histogram(f'Histograma imagen {i}', gray1)
hist2 = histogram(f'Histograma imagen {j}', gray2)
hist3 = histogram(f'Histograma imagen {k}', gray3)

plt.show()

cv.waitKey(0)