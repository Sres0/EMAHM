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

def cvt_n_split(img, method=cv.COLOR_BGR2HSV):
    img = cv.cvtColor(img, method)
    x, y, z = cv.split(img)
    return img, [x, y, z]

gelGray = cv.cvtColor(imgGel, cv.COLOR_BGR2GRAY)
lBgr = cv.split(imgGel)
hsv, lHsv = cvt_n_split(imgGel)
lab, lLab = cvt_n_split(imgGel, cv.COLOR_BGR2LAB)

### 1D Histogram ###
def one_d_histogram(imgChannels):
    lColors = ('b', 'g', 'r')
    plt.figure()
    plt.title('Histograma 1D')
    plt.xlabel('Bins')
    plt.ylabel('Pixeles')
    plt.grid()

    # for chan in imgChannels:
    hist = cv.calcHist([imgChannels], [0], None, [255], [0, 256])
    plt.plot(hist)
    # plt.plot(hist, color=col)
    plt.xlim([0, 256])

one_d_histogram(lHsv[0])

### 2D Histogram ###
# hist = cv.calcHist([h, s], [0, 1], None, [255, 255], [0, 256, 0, 256])
# fig = plt.figure()
# ax = fig.add_subplot(131)
# p = ax.imshow(hist, interpolation='nearest')
# ax.set_title('dsfd')
# plt.colorbar(p)
# plt.show()

### Equalize ###
equ = cv.equalizeHist(lHsv[1])
res = np.hstack((lHsv[1], equ))
cv.imshow('image', res)
one_d_histogram(equ)

h_hand_mask_gel = threshold(i, 135, lHsv[0], f'hue gel')
eq_h_hand_mask_gel = threshold(i, 80, equ, f'hue ecualizado gel')
# g_hand_mask_gel = threshold(i, 75, g, f'verde original gel')

plt.show()
cv.waitKey()