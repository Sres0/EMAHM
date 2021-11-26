import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

i = 510 # con gel
j = i - 2

def path(i):
    return f'Processing/Bank/Imgs/hand_test_{i}.png'

def createImg(i, hide=True, resized=False):
    img = cv.imread(path(i))
    if not resized:
        w = int(1280/3)
        h = int(720/3)
        img = cv.resize(img, (w,h))
    if not hide: cv.imshow(f'Original {i}', img)
    return img

handGel = createImg(i)
handNoGel = createImg(j)
# HandGel = createImg(i, hide=False)
# HandNoGel = createImg(j, hide=False)
# cv.imshow('Segmented gel & no gel', np.hstack([HandGel, HandNoGel]))

### MASK ###

medium_blur = (3,3)
big_blur = (5,5)

blurredHandGel = cv.blur(handGel, medium_blur)
blurredHandNoGel = cv.blur(handNoGel, medium_blur)

grayBlurredHandGel = cv.cvtColor(blurredHandGel, cv.COLOR_BGR2GRAY)
grayBlurredHandNoGel = cv.cvtColor(blurredHandNoGel, cv.COLOR_BGR2GRAY)

def one_d_histogram(img, title, hide=True):

    if not hide:
        plt.figure()
        plt.title(title)
        plt.xlabel('Bins')
        plt.ylabel('Pixels')
        plt.grid()
        plt.xlim([0, 256])

    if type(img) == list:
        for cnl in img:
            hist = cv.calcHist([cnl], [0], None, [255], [0, 256])
            if not hide: plt.plot(hist)
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
    thresh = next(filter(lambda index: index > 75, min_peaks), None)
    if not hide:
        plt.figure()
        plt.title(title + f' | {thresh}')
        plt.grid()
        plt.plot(x,hist)
        plt.plot(x,y, color='red')
        plt.plot(thresh,y[thresh], "x")
        plt.xlim([0, 256])

    return thresh

hGrayBlurredHandGel = one_d_histogram(grayBlurredHandGel, 'Blurred gray gel', hide=True)
hGrayBlurredHandNoGel = one_d_histogram(grayBlurredHandNoGel, 'Blurred gray no gel', hide=True)

def threshold(i, mval, img, channel_name, method=cv.THRESH_BINARY, hide=True):
    _, thresh = cv.threshold(img, mval, 255, method)
    if not hide: cv.imshow(f'Mascara {channel_name} {i} | {mval}', thresh)
    return thresh

mGrayBlurredHandGel = threshold(i, find_hand_thresh(hGrayBlurredHandGel, title='gel thresh', hide=True), grayBlurredHandGel, 'gray blurred gel')
mGrayBlurredHandNoGel = threshold(i, find_hand_thresh(hGrayBlurredHandNoGel, title='no gel thresh', hide=True), grayBlurredHandNoGel, 'gray blurred no gel')
# cv.imshow('Segmented gel & no gel mask', np.hstack([mGrayBlurredHandGel, mGrayBlurredHandNoGel]))

### PENDING MORPH

def contour(mask, img, color, i, title, hide=False):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 1000]
    for cnt in contours: cv.drawContours(img, [cnt], -1, color, 1)
    if not hide: cv.imshow(f'contorno {title} {i}', img)
    return [cv.contourArea(cnt) for cnt in contours]

copyHandGel = createImg(i)
copyHandNoGel = createImg(j)
aHandGel = contour(mGrayBlurredHandGel, copyHandGel, (255, 255, 255), i, 'gray blurred gel', hide=True)
aHandNoGel = contour(mGrayBlurredHandNoGel, copyHandNoGel, (255, 255, 255), j, 'gray blurred no gel', hide=True)
# cv.imshow('Segmented gel & no gel contour', np.hstack([copyHandGel, copyHandNoGel]))

### Gel ###

hGel,sGel,vGel = cv.split(cv.cvtColor(handGel, cv.COLOR_BGR2HSV))
hsvGel = [hGel,sGel,vGel]
hNoGel,sNoGel,vNoGel = cv.split(cv.cvtColor(handNoGel, cv.COLOR_BGR2HSV))
hsvNoGel = [hNoGel,sNoGel,vNoGel]

one_d_histogram(hsvGel, 'HSV Gel', hide=False)
one_d_histogram(hsvNoGel, 'HSV no Gel', hide=False)

plt.show()
cv.waitKey(0)