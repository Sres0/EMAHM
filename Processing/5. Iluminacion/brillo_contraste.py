import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 76 # con gel

def path(i):
    return f'Processing/Post/Bank/Imgs/hand_test_{i}.png'

def createImg(path, i, hide=False, write=False):
    img = cv.imread(path)
    w = int(1280/3)
    h = int(720/3)
    resized = cv.resize(img, (w,h))
    if not hide: cv.imshow(f'Original {i}', resized)
    if write: cv.imwrite(f'Processing/Post/Processing/Test/Documentation/5. Iluminacion/original_{i}.png', resized)
    return resized

img = createImg(path(i), i)
hsv = createImg(path(i), i, 1)
hsvEdited = createImg(path(i), i, 1)

### Máscara ###

def threshold(i, mval, img, channel_name, method=cv.THRESH_BINARY):
    _, thresh = cv.threshold(img, mval, 255, method)
    # cv.imshow(f'Mascara {channel_name} {i} | {mval}', thresh)
    return thresh

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hand_mask = threshold(i, 105, gray, 'gris')

### Brillo y contraste ###

alpha = 1 # Contrast 1
beta = -100 # Brightness -100
hsvEdited = cv.convertScaleAbs(hsvEdited, alpha=alpha, beta=beta)

cv.imshow(f'hsv editada brillo y contraste {i}', hsvEdited)

### Canales ###

def show_channels(imgs, titles, j):
    for i in range(len(imgs)):
        # cv.imshow(titles[i], imgs[i])
        pass

def split_channels(img, cvt, titles, i):
    img = cv.cvtColor(img, cvt)
    img = cv.bitwise_and(img, img, mask=hand_mask)
    x, y, z = cv.split(img)
    show_channels([x,y,z], titles, i)
    return img, x, y, z

channel_names = ['hue', 'saturation', 'value', 'hue_editada', 'saturation_editada', 'value_editada']
hsv, h, s, v = split_channels(hsv, cv.COLOR_BGR2HSV, [channel_names[i] for i in range(0, 3)], i)
cv.imshow('HSV', hsv)
hsvEdited, hE, sE, vE = split_channels(hsvEdited, cv.COLOR_BGR2HSV, [channel_names[i] for i in range(3, 6)], i)
cv.imshow('HSV editada', hsvEdited)
channel_imgs = [h, s, v, hE, sE, vE]

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

### Contornos ###

def contour(mask, img, color, i, channel_name, write=False):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    cv.imshow(f'contorno {channel_name} {i}', img)
    if write: cv.imwrite(f'Processing/Post/Processing/Test/Documentation/5. Iluminacion/contour_{channel_name}_{i}.png', img)

lower_hsv = np.array([100, 5, 150], dtype="uint8") # 100, 5, 150
upper_hsv = np.array([150, 100, 255], dtype="uint8") # 150, 100, 255
lower_hsv_edited = np.array([100, 5, 110], dtype="uint8") # 100, 5, 110
upper_hsv_edited = np.array([150, 180, 255], dtype="uint8") # 150, 180, 255

hsv_mask = cv.inRange(hsv, lower_hsv, upper_hsv)
hsv_edited_mask = cv.inRange(hsvEdited, lower_hsv_edited, upper_hsv_edited)

img2 = createImg(path(i), i)
contour(hsv_mask, img, (255, 0, 255), i, 'HSV')
contour(hsv_edited_mask, img2, (255, 255, 0), i, 'HSV editada')

plt.show()
cv.waitKey()