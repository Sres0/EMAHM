import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 76 # con gel
j = i - 2 # sin gel

def path(i):
    return f'Processing/Bank/Imgs/hand_test_{i}.png'

def createImg(path, i, hide=False):
    img = cv.imread(path)
    w = int(1280/3)
    h = int(720/3)
    resized = cv.resize(img, (w,h))
    if not hide: cv.imshow(f'Original {i}', resized)
    return resized

imgGel = createImg(path(i), i)
imgNoGel = createImg(path(j), j)
hsv = createImg(path(i), i, 1)
lab = createImg(path(i), i, 1)
bgr = createImg(path(i), i, 1)

def show_channels(imgs, titles, j):
    for i in range(len(imgs)):
        cv.imshow(titles[i], imgs[i])
        pass

### Máscara ###

def threshold(i, mval, img, channel_name, method=cv.THRESH_BINARY):
    _, thresh = cv.threshold(img, mval, 255, method)
    cv.imshow(f'Mascara {channel_name} {i} | {mval}', thresh)
    return thresh

gray = cv.cvtColor(imgGel, cv.COLOR_BGR2GRAY)
hand_mask = threshold(i, 105, gray, 'gris')

### BGR, HSV & LAB ###

def split_channels(img, cvt, titles, i):
    img = cv.cvtColor(img, cvt)
    img = cv.bitwise_and(img, img, mask=hand_mask)
    x, y, z = cv.split(img)
    show_channels([x,y,z], titles, i)
    return img, x, y, z

channel_names = ['hue', 'saturation', 'value', 'lightness', 'a', 'b', 'azul', 'verde', 'rojo']
hsv, h, s, v = split_channels(hsv, cv.COLOR_BGR2HSV, [channel_names[i] for i in range(0, 3)], i)
lab, l, a, b = split_channels(lab, cv.COLOR_BGR2LAB, [channel_names[i] for i in range(3, 6)], i)
bgr, bl, g, r = split_channels(bgr, cv.COLOR_BGR2LAB, [channel_names[i] for i in range(6, 9)], i)
channel_imgs = [h, s, v, l, a, b, bl, g, r]

### Histograma ###

def histogram(i, img, channel_name):
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    plt.figure()
    plt.title(f'Histograma {channel_name} imagen {i}')
    plt.xlabel('Intensidad de pixel (0 - 255)')
    plt.ylabel('Número de pixeles')
    plt.grid()
    plt.plot(hist)
    plt.xlim([0,256])

for j in range(len(channel_imgs)):
    histogram(i, channel_imgs[j], channel_names[j])

def contour(mask, img, color, i, channel_name):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    cv.imshow(f'contorno {channel_name} {i}', img)

lowerHSV = np.array([100, 5, 150], dtype="uint8") # 100, 5, 150
upperHSV = np.array([150, 100, 255], dtype="uint8") # 150, 100, 255
lowerLAB = np.array([160, 125, 90], dtype="uint8") # 100, 125, 60
upperLAB = np.array([250, 165, 130], dtype="uint8") # 250, 180, 130

hsv_mask = cv.inRange(hsv, lowerHSV, upperHSV)
lab_mask = cv.inRange(lab, lowerLAB, upperLAB)
green_mask = threshold(i, 160, g, 'green', cv.THRESH_BINARY_INV)
green_mask = cv.bitwise_and(green_mask, green_mask, mask=hand_mask)

imgGel2 = createImg(path(i), i)
imgGel3 = createImg(path(i), i)
contour(hsv_mask, imgGel, (0, 255, 255), i, 'hsv')
contour(lab_mask, imgGel2, (255, 0, 255), i, 'lab')
contour(green_mask, imgGel3, (0, 255, 0), i, 'verde')

plt.show()

cv.waitKey()