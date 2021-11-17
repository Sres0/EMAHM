import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 80 # con gel
j = i - 2

def path(i):
    return f'Processing/Bank/Imgs/hand_test_{i}.png'

def createImg(i, hide=False):
    img = cv.imread(path(i))
    w = int(1280/3)
    h = int(720/3)
    resized = cv.resize(img, (w,h))
    if not hide: cv.imshow(f'Original {i}', resized)
    return resized

imgGel = createImg(i)
# imgNoGel = createImg(j)

### Máscara mano ###

def threshold(i, mval, img, channel_name, method=cv.THRESH_BINARY):
    _, thresh = cv.threshold(img, mval, 255, method)
    cv.imshow(f'Mascara {channel_name} {i} | {mval}', thresh)
    return thresh

def histogram(i, img, channel_name):
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    plt.figure()
    plt.title(f'Histograma {channel_name} imagen {i}')
    plt.xlabel('Intensidad de pixel (0 - 255)')
    plt.ylabel('Número de pixeles')
    plt.grid()
    plt.plot(hist)
    plt.xlim([0,256])

gelGray = cv.cvtColor(imgGel, cv.COLOR_BGR2GRAY)
b, g, r = cv.split(imgGel)
h, s, v = cv.split(cv.cvtColor(imgGel, cv.COLOR_BGR2HSV))
channel_names = ['Gray', 'Green']
channel_imgs = [gelGray, g]

for j in range(len(channel_imgs)):
    cv.imshow(channel_names[j], channel_imgs[j])
    histogram(i, channel_imgs[j], channel_names[j])

gray_hand_mask_gel = threshold(i, 112, gelGray, f'gris original gel')
g_hand_mask_gel = threshold(i, 75, g, f'verde original gel')

plt.show()
cv.waitKey()