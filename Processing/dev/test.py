import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 63 # con gel
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

small_blur = (2,2)
medium_blur = (5,5)

gelGray = cv.cvtColor(imgGel, cv.COLOR_BGR2GRAY)
lBgr = cv.split(imgGel)

### 1D Histogram ###
def one_d_histogram(img):
    lColors = ('b', 'g', 'r')
    plt.figure()
    plt.title('Histograma 1D')
    plt.xlabel('Bins')
    plt.ylabel('Pixeles')
    plt.grid()

    if type(img) == list:
        for (chan, col) in zip(img, lColors):
            hist = cv.calcHist([chan], [0], None, [256], [0, 256])
            plt.plot(hist, color=col)
    else:
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist)
    
    plt.xlim([0, 256])
    indices = list(range(0, 255))
    hist = [y[0] for y in hist]
    # hist = [(x,y[0]) for x,y in zip(indices,hist)]
    return hist

hGelGray = one_d_histogram(gelGray)

from scipy.signal import savgol_filter
# print(hGelGray[:])
plt.figure()

x = np.array(range(0,256))
y = savgol_filter(hGelGray[:], 35, 3)

plt.grid()
plt.plot(x,hGelGray[:])
plt.plot(x,y, color='red')
plt.xlim([0, 256])
plt.show()

# print(hGelGray[254][1])

# ### 2D Histogram ###
# # hist = cv.calcHist([h, s], [0, 1], None, [255, 255], [0, 256, 0, 256])
# # fig = plt.figure()
# # ax = fig.add_subplot(131)
# # p = ax.imshow(hist, interpolation='nearest')
# # ax.set_title('dsfd')
# # plt.colorbar(p)
# # plt.show()

# ### Equalize ###
# equ = cv.equalizeHist(lHsv[1])
# res = np.hstack((lHsv[1], equ))
# cv.imshow('image', res)
# one_d_histogram(equ)

# h_hand_mask_gel = threshold(i, 135, lHsv[0], f'hue gel')
# eq_h_hand_mask_gel = threshold(i, 80, equ, f'hue ecualizado gel')
# g_hand_mask_gel = threshold(i, 75, g, f'verde original gel')

# plt.show()
cv.waitKey()