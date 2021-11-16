import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 76 # con gel
j = i - 2 # sin gel

def path(i):
    return f'Processing/Post/Bank/Imgs/hand_test_{i}.png'

def createImg(path, i, show=False):
    img = cv.imread(path)
    w = int(1280/3)
    h = int(720/3)
    resized = cv.resize(img, (w,h))
    if not show: cv.imshow(f'Original {i}', resized)
    # cv.imwrite(f'Processing/Post/Processing/Test/Documentation/2. Diagramas/original_{i}.png', resized)
    return resized

imgGel = createImg(path(i), i)
imgNoGel = createImg(path(j), j)
hsv = createImg(path(i), i, 1)
lab = createImg(path(i), i, 1)

def show_channels(imgs, titles, j):
    for i in range(len(imgs)):
        cv.imshow(titles[i], imgs[i])
        # cv.imwrite(f'Processing/Post/Processing/Test/Documentation/3. Canales/{titles[i]}_{j}.png', imgs[i])

### Máscara ###

def threshold(i, mval, img, channel):
    _, thresh = cv.threshold(img, mval, 255, cv.THRESH_BINARY)
    cv.imshow(f'Mascara {channel} {i} | {mval}', thresh)
    # cv.imwrite(f'Processing/Post/Processing/Test/Documentation/3. Canales/threshold_{channel}_{i}_{mval}.png', thresh)
    return thresh

gray = cv.cvtColor(imgGel, cv.COLOR_BGR2GRAY)
hand_mask = threshold(i, 105, gray, 'gris')

### HSV & LAB ###

def split_channels(img, cvt, titles, i):
    img = cv.cvtColor(img, cvt)
    img = cv.bitwise_and(img, img, mask=hand_mask)
    x, y, z = cv.split(img)
    show_channels([x,y,z], titles, i)
    return img, x, y, z

channel_names = ['hue', 'saturation', 'value', 'lightness', 'a', 'b']
hsv, h, s, v = split_channels(hsv, cv.COLOR_BGR2HSV, [channel_names[i] for i in range(0, 3)], i)
lab, l, a, b = split_channels(lab, cv.COLOR_BGR2LAB, [channel_names[i] for i in range(3, 6)], i)
channel_imgs = [h, s, v, l, a, b]

### Histograma ###

def channel_str(channel):
    if channel == 'h' or channel == 'b': channel_str = 'azul'
    elif channel == (0, 255, 0) or channel == 'g': channel_str = 'verde'
    elif channel == (0, 0, 255) or channel == 'r': channel_str = 'rojo'
    else: channel_str = ''
    return channel_str

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

def contour(mask, img, color, i, channel):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    cv.imshow(f'contorno {channel} {i}', img)
    # cv.imwrite(f'Processing/Post/Processing/Test/Documentation/4. Espacios/contour_{channel}_{i}.png', img)

# bHist = histogram(i, b, 'b', channel_str)
# gHist = histogram(i, g, 'g', channel_str)
# rHist = histogram(i, r, 'r', channel_str)

# bThresh = threshold(i, 200, b, channel_str, 'b')
# gThresh = threshold(i, 60, g, channel_str, 'g')
# rThresh = threshold(i, 125, r, channel_str, 'r')

# contour(bThresh, img1, (255, 0, 0), i, channel_str)
# contour(gThresh, img2, (0, 255, 0), i, channel_str)
# contour(rThresh, img3, (0, 0, 255), i, channel_str)

plt.show()

cv.waitKey()