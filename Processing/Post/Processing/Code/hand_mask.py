import cv2 as cv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

i = 0
j = 0

def readOrWrite(i, j, bank):
    if bank == 1:
        return 'Processing/Post/Bank/Imgs/hand_test_'+str(i)+'.png'
    else:
        return 'Processing/Post/Processing/Registry/4. Channels/4.'+str(j)+'. threshold_'+str(i)+'_green.png'

plt.figure()

def plot(name, j, title):
    pltimg = mpimg.imread(name)
    plt.subplot(4,4,j+1)
    plt.axis('off')
    plt.title(title)
    plt.imshow(pltimg, cmap='gray')
    return j+1

for i in range(4):
    img = cv.imread(readOrWrite(i, j, 1))
    w = int(1280/3)
    h = int(720/3)
    img = cv.resize(img, (w,h))
    j = plot(readOrWrite(i, j, 1), j, 'Mano '+str(i))
    # print(j)

    b,g,r = cv.split(img)
    # # cv.imshow('Green', g)
    # # cv.imwrite('Processing/Post/Processing/Registry/4. Channels/0.0. hand_test_0_green.png', g)

    ### THRESH ###
    blank = np.zeros(img.shape, dtype='uint8')
    ret, g_thresh = cv.threshold(g, 75, 255, cv.THRESH_BINARY)
    # cv.imwrite(readOrWrite(i, j, 0), g_thresh)
    j = plot(readOrWrite(i, j, 0), j, 'verde 75')
    ret, r_thresh = cv.threshold(r, 100, 255, cv.THRESH_BINARY)
    # cv.imwrite(readOrWrite(i, j, 0), r_thresh)
    j = plot(readOrWrite(i, j, 0), j, 'rojo 100')
    ret, b_thresh = cv.threshold(b, 175, 255, cv.THRESH_BINARY)
    # cv.imwrite(readOrWrite(i, j, 0), b_thresh)
    j = plot(readOrWrite(i, j, 0), j, 'azul 175')

    i += 1

plt.show()

cv.waitKey(0)