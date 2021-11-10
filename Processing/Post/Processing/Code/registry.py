import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Processing/Post/Bank/Imgs/hand_test_0.png')
pltimg = mpimg.imread('Processing/Post/Bank/Imgs/hand_test_0.png')

def split(img, name):
    # h, w, _ = img.shape # _ channels
    # b,g,r = cv.split(img)
    i = 14
    plt.figure()
    for i in range(3):
        plt.subplot(2,3,i+1)
        plt.imshow(pltimg[:,:,i], cmap='gray')
        i += 1

split(pltimg, 'Figure 1')

### HISTOGRAMS ###
b_hist = cv.calcHist([img], [0], None, [256], [0,256])
g_hist = cv.calcHist([img], [1], None, [256], [0,256])
r_hist = cv.calcHist([img], [2], None, [256], [0,256])

def grid(hist, i, name):
    plt.subplot(2,3,i)
    plt.grid()
    plt.title(f'Histograma {name}')
    plt.xlabel('Intensidad')
    plt.ylabel('Pixeles')
    plt.ylim([0,50000])
    plt.xlim([0,256])
    plt.plot(hist)

grid(r_hist, 4, 'rojo')
grid(g_hist, 5, 'verde')
grid(b_hist, 6, 'azul')
plt.show()

cv.waitKey(0)