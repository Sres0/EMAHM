import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

i = 68 # con gel
j = i - 2

def path(i):
    return f'Processing/Bank/Imgs/hand_test_{i}.png'

def createImg(i, hide=True):
    img = cv.imread(path(i))
    w = int(1280/3)
    h = int(720/3)
    resized = cv.resize(img, (w,h))
    if not hide: cv.imshow(f'Original {i}', resized)
    return resized

imgGel = createImg(i)
imgNoGel = createImg(j)
eqImgNoGel = createImg(j)
# imgGel = createImg(i, hide=False)
# imgNoGel = createImg(j, hide=False)
# cv.imshow('Segmented gel & no gel', np.hstack([imgGel, imgNoGel]))

### Mask ###

medium_blur = (3,3)

blurredImgGel = cv.blur(imgGel, medium_blur)
blurredImgNoGel = cv.blur(imgNoGel, medium_blur)

grayBlurredImgGel = cv.blur(cv.cvtColor(blurredImgGel, cv.COLOR_BGR2GRAY), medium_blur)
grayBlurredImgNoGel = cv.blur(cv.cvtColor(blurredImgNoGel, cv.COLOR_BGR2GRAY), medium_blur)
grayEqBlurredImgNoGel = cv.equalizeHist(grayBlurredImgNoGel)

def one_d_histogram(img, title, hide=True):

    if not hide:
        plt.figure()
        plt.title(title)
        plt.xlabel('Bins')
        plt.ylabel('Pixels')
        plt.grid()
        plt.xlim([0, 256])

    if type(img) == list:
        for chan in img:
            hist = cv.calcHist([chan], [0], None, [255], [0, 256])
    else:
        hist = cv.calcHist([img], [0], None, [255], [0, 256])
    if not hide: plt.plot(hist)
    hist = [y[0] for y in hist]

    return hist

def find_hand_thresh(hist, fs=60, hide=True, title='hand threshold'):

    # outliers
    y = np.where(hist > (np.mean(hist) + 10*np.std(hist)), np.mean(hist), hist)
    y = np.where(hist < (np.mean(hist) - 10*np.std(hist)), np.mean(hist), hist)

    # butter
    wn = 8/(fs/2)
    b, a = sig.butter(3, wn, btype='lowpass')
    y = sig.filtfilt(b, a, y)
    x = np.array(range(0,255))

    # peaks
    min_peaks, _ = sig.find_peaks(-y, height=(-500, -1))
    thresh = next(filter(lambda index: index > 50, min_peaks), None)
    if not hide:
        plt.figure()
        plt.title(title + f' | {thresh}')
        plt.grid()
        plt.plot(x,hist)
        plt.plot(x,y, color='red')
        plt.plot(thresh,y[thresh], "x")
        plt.xlim([0, 256])

    return thresh

hGrayBlurredImgGel = one_d_histogram(grayBlurredImgGel, 'Blurred gray gel', hide=True)
hGrayBlurredImgNoGel = one_d_histogram(grayBlurredImgNoGel, 'Blurred gray no gel', hide=True)
hGrayEqBlurredImgNoGel = one_d_histogram(grayEqBlurredImgNoGel, 'Equalized blurred gray no gel', hide=False)

def threshold(i, mval, img, channel_name, method=cv.THRESH_BINARY, hide=True):
    _, thresh = cv.threshold(img, mval, 255, method)
    if not hide: cv.imshow(f'Mascara {channel_name} {i} | {mval}', thresh)
    return thresh

mGrayBlurredImgGel = threshold(i, find_hand_thresh(hGrayBlurredImgGel, title='gel thresh', hide=True), grayBlurredImgGel, 'gray blurred gel')
mGrayBlurredImgNoGel = threshold(i, find_hand_thresh(hGrayBlurredImgNoGel, title='no gel thresh', hide=True), grayBlurredImgNoGel, 'gray blurred no gel')
mGrayEqBlurredImgNoGel = threshold(i, 185, grayEqBlurredImgNoGel, 'gray equalized blurred no gel')
# cv.imshow('Segmented gel & no gel mask', np.hstack([mGrayBlurredImgGel, mGrayBlurredImgNoGel]))

mGrayBlurredImgGel = cv.erode(mGrayBlurredImgGel, np.array((2,2), dtype='uint8'), iterations=1)
mGrayBlurredImgNoGel = cv.erode(mGrayBlurredImgNoGel, np.array((2,2), dtype='uint8'), iterations=1)
mGrayEqBlurredImgNoGel = cv.erode(mGrayEqBlurredImgNoGel, np.array((2,2), dtype='uint8'), iterations=1)

def contour(mask, img, color, i, title, hide=False):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    if not hide: cv.imshow(f'contorno {title} {i}', img)
    return contours

contour(mGrayBlurredImgGel, imgGel, (255, 255, 255), i, 'gray blurred gel', 1)
contour(mGrayBlurredImgNoGel, imgNoGel, (255, 255, 255), i, 'gray blurred no gel', 1)
contour(mGrayEqBlurredImgNoGel, eqImgNoGel, (255, 255, 255), i, 'gray blurred no gel', 1)
cv.imshow('Segmented gel & no gel contour', np.hstack([imgGel, imgNoGel, eqImgNoGel]))

### Gel ###

# def cvt_n_split(img, method=cv.COLOR_BGR2HSV):
#     img = cv.cvtColor(img, method)
#     x, y, z = cv.split(img)
#     return [x, y, z, img]

plt.show()
cv.waitKey(0)