import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 73 # con gel
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

# imgGel = createImg(i)
# imgNoGel = createImg(j)
imgGel = createImg(i, 1)
imgNoGel = createImg(j, 1)
# cv.imshow('Segmented gel & no gel', np.hstack([imgGel, imgNoGel]))

### BLUR ###

medium_blur = (3,3)

blurredImgGel = cv.blur(imgGel, medium_blur)
blurredImgNoGel = cv.blur(imgNoGel, medium_blur)
grayBlurredImgGel = cv.blur(cv.cvtColor(blurredImgGel, cv.COLOR_BGR2GRAY), medium_blur)
grayBlurredImgNoGel = cv.blur(cv.cvtColor(blurredImgNoGel, cv.COLOR_BGR2GRAY), medium_blur)

def one_d_histogram(img, title):
    lColors = ('b', 'g', 'r')
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('Pixels')
    plt.grid()

    if type(img) == list:
        for (chan, col) in zip(img, lColors):
            hist = cv.calcHist([chan], [0], None, [255], [0, 256])
            plt.plot(hist, color=col)
    else:
        hist = cv.calcHist([img], [0], None, [255], [0, 256])
        plt.plot(hist)
    plt.xlim([0, 256])

# one_d_histogram(blurredImgGel, 'Blurred gel')
one_d_histogram(grayBlurredImgGel, 'Blurred gray gel')
one_d_histogram(grayBlurredImgNoGel, 'Blurred gray no gel')

def threshold(i, mval, img, channel_name, method=cv.THRESH_BINARY, hide=False):
    _, thresh = cv.threshold(img, mval, 255, method)
    if not hide: cv.imshow(f'Mascara {channel_name} {i} | {mval}', thresh)
    return thresh

mGrayBlurredImgGel = threshold(i, 81.5, grayBlurredImgGel, 'gray blurqred gel', hide=True)
mGrayBlurredImgNoGel = threshold(i, 81.5, grayBlurredImgNoGel, 'gray blurred no gel', hide=False)
# cv.imshow('Segmented gel & no gel', np.hstack([mGrayBlurredImgGel, mGrayBlurredImgNoGel]))

def contour(mask, img, color, i, title, hide=False):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    if not hide: cv.imshow(f'contorno {title} {i}', img)
    return contours

contour(mGrayBlurredImgGel, imgGel, (255, 255, 255), i, 'gray blurred gel', 1)
contour(mGrayBlurredImgNoGel, imgNoGel, (255, 255, 255), i, 'gray blurred no gel', 1)
cv.imshow('Segmented gel & no gel', np.hstack([imgGel, imgNoGel]))

# def cvt_n_split(img, method=cv.COLOR_BGR2HSV):
#     img = cv.cvtColor(img, method)
#     x, y, z = cv.split(img)
#     return [x, y, z, img]

plt.show()
cv.waitKey(0)