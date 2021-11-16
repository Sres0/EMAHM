import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 76

def path(i):
    return f'Processing/Bank/Imgs/hand_test_{i}.png'

def createImg(path, i):
    img = cv.imread(path)
    w = int(1280/3)
    h = int(720/3)
    resized = cv.resize(img, (w,h))
    cv.imshow(f'Original {i}', resized)
    return resized

path1 = path(i)

img1 = createImg(path1, i)
img2 = createImg(path1, i)
img3 = createImg(path1, i)

def show_channels(imgs, titles, j):
    for i in range(len(imgs)):
        cv.imshow(titles[i], imgs[i])

b,g,r = cv.split(img1)
show_channels([b,g,r], ['azul', 'verde', 'rojo'], i)

def histogram(i, img, channel=False, channel_str=''):
    if channel:
        channel_str = channel_str(channel)
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    plt.figure()
    plt.title(f'Histograma {channel_str} imagen {i}')
    plt.xlabel('Intensidad de pixel (0 - 255)')
    plt.ylabel('Número de pixeles')
    plt.grid()
    plt.plot(hist)
    plt.xlim([0,256])

def threshold(i, mval, img, channel_str, channel):
    _, thresh = cv.threshold(img, mval, 255, cv.THRESH_BINARY)
    channel_str = channel_str(channel)
    cv.imshow(f'Mascara {channel_str} {i} | {mval}', thresh)
    return thresh

def channel_str(channel):
    if channel == (255, 0, 0) or channel == 'b': channel_str = 'azul'
    elif channel == (0, 255, 0) or channel == 'g': channel_str = 'verde'
    elif channel == (0, 0, 255) or channel == 'r': channel_str = 'rojo'
    else: channel_str = ''
    return channel_str

def contour(mask, img, color, i, channel_str):
    channel_str = channel_str(color)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    cv.imshow(f'contorno {channel_str} {i}', img)

bHist = histogram(i, b, 'b', channel_str)
gHist = histogram(i, g, 'g', channel_str)
rHist = histogram(i, r, 'r', channel_str)

bThresh = threshold(i, 200, b, channel_str, 'b')
gThresh = threshold(i, 60, g, channel_str, 'g')
rThresh = threshold(i, 125, r, channel_str, 'r')

contour(bThresh, img1, (255, 0, 0), i, channel_str)
contour(gThresh, img2, (0, 255, 0), i, channel_str)
contour(rThresh, img3, (0, 0, 255), i, channel_str)

plt.show()

cv.waitKey()