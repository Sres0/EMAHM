import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

i = 630 # con gel
j = i - 2

def path(i):
    # return f'Processing/reporte_final/{i}.png'
    return f'Processing/Bank/Imgs/hand_test_{i}.png'

def createImg(i, hide=True, resized=False):
    img = cv.imread(path(i))
    if not resized:
        w = int(1280/3)
        h = int(720/3)
        img = cv.resize(img, (w,h))
        img = img[0:int(h*4/5), 0:w] # recortar imagen en muñeca
    if not hide: cv.imshow(f'Original {i}', img)
    return img

handGel = createImg(i, hide=True)
handNoGel = createImg(j, hide=True)
# cv.imshow('gel & no gel', np.hstack([handGel, handNoGel]))

### MASK ###

medium_blur = (3,3)
big_blur = (5,5)

blurredHandGel = cv.blur(handGel, medium_blur)
blurredHandNoGel = cv.blur(handNoGel, medium_blur)

grayBlurredHandGel = cv.cvtColor(blurredHandGel, cv.COLOR_BGR2GRAY)
grayBlurredHandNoGel = cv.cvtColor(blurredHandNoGel, cv.COLOR_BGR2GRAY)

def set_histogram(title):
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('Pixels')
    plt.grid()
    plt.autoscale()
    # plt.xlim([100, 150])
    # plt.ylim([0, 5000])

def one_d_histogram(img, title, hide=True, min=0, normalize=False):
    if not hide: set_histogram(title)
    
    if type(img) == list:
        for cnl in img:
            hist = cv.calcHist([cnl], [0], None, [255], [min, 256])
            if normalize: cv.normalize(hist,hist,0,1,cv.NORM_MINMAX)
            if not hide: plt.plot(hist)
    else:
        hist = cv.calcHist([img], [0], None, [255], [min, 256])
        if normalize: cv.normalize(hist,hist,0,1,cv.NORM_MINMAX)
        if not hide: plt.plot(hist)

    hist = [y[0] for y in hist] # Retorna el histograma del último canal

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
        set_histogram(title + f' | {thresh}')
        plt.plot(x,hist)
        plt.plot(x,y, color='red')
        plt.plot(thresh,y[thresh], "x")
    return thresh

hGrayBlurredHandGel = one_d_histogram(grayBlurredHandGel, 'Blurred gray gel', hide=True)
hGrayBlurredHandNoGel = one_d_histogram(grayBlurredHandNoGel, 'Blurred gray no gel', hide=True)

def get_binary_mask(i, thresh, img, channel_name, method=cv.THRESH_BINARY, hide=True):
    _, mask = cv.threshold(img, thresh, 255, method)
    if not hide: cv.imshow(f'Mascara {channel_name} {i} | {thresh}', mask)
    return mask

mGrayBlurredHandGel = get_binary_mask(i, find_hand_thresh(hGrayBlurredHandGel, title='gel thresh', hide=False), grayBlurredHandGel, 'gray blurred gel')
mGrayBlurredHandNoGel = get_binary_mask(i, find_hand_thresh(hGrayBlurredHandNoGel, title='no gel thresh', hide=False), grayBlurredHandNoGel, 'gray blurred no gel')
# cv.imshow('Segmented gel & no gel mask', np.hstack([mGrayBlurredHandGel, mGrayBlurredHandNoGel]))

### PENDING MORPH

def contour(mask, img, color, i, title, hide=True):
    blank = np.zeros((img.shape), dtype=np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 1000]
    for cnt in contours:
        cv.drawContours(img, [cnt], -1, color, 1)
        cv.drawContours(blank, [cnt], -1, color, -1)
    if not hide: cv.imshow(title+f' {i}', np.hstack([img, blank]))
    return blank, [cv.contourArea(cnt) for cnt in contours], contours

copyHandGel = handGel.copy()
copyHandNoGel = handNoGel.copy()
mHandGel, aHandGel, gelContours = contour(mGrayBlurredHandGel, copyHandGel, (255, 255, 255), i, 'gray blurred gel', hide=True)
mHandNoGel, aHandNoGel, noGelContours = contour(mGrayBlurredHandNoGel, copyHandNoGel, (255, 255, 255), j, 'gray blurred no gel', hide=True)
# cv.imshow('Segmented gel & no gel contour', np.hstack([copyHandGel, copyHandNoGel]))

### Gel ###
# # HSV #
# imgHsvGel = cv.cvtColor(cv.bitwise_and(handGel, mHandGel), cv.COLOR_BGR2HSV)
# hGel,sGel,vGel = cv.split(imgHsvGel)
# imgHsvNoGel = cv.cvtColor(cv.bitwise_and(handNoGel, mHandNoGel), cv.COLOR_BGR2HSV)
# hNoGel,sNoGel,vNoGel = cv.split(imgHsvNoGel)
# # cv.imshow('hsv', np.hstack([imgHsvGel, imgHsvNoGel]))

# hGelVsNoGel = [hGel,hNoGel]
# sGelVsNoGel = [sGel,sNoGel]
# vGelVsNoGel = [vGel,vNoGel]
# hHGelVsNoGel = one_d_histogram(hGelVsNoGel, f'Hue gel vs no gel | {i}', min=1, hide=False)
# hSGelVsNoGel = one_d_histogram(sGelVsNoGel, f'Sat gel vs no gel | {i}', min=1, hide=False)
# hVGelVsNoGel = one_d_histogram(vGelVsNoGel, f'Val gel vs no gel | {i}', min=1, hide=False)

# LAB #
imgLabGel = cv.bitwise_and(cv.cvtColor(handGel, cv.COLOR_BGR2LAB), mHandGel)
lGel,aGel,bGel = cv.split(imgLabGel)
imgLabNoGel = cv.cvtColor(cv.bitwise_and(handNoGel, mHandNoGel), cv.COLOR_BGR2HSV)
lNoGel,aNoGel,bNoGel = cv.split(imgLabNoGel)
# cv.imshow('hsv', np.hstack([imgLabGel, imgLabNoGel]))

# lGelVsNoGel = [lGel,hNoGel]
# aGelVsNoGel = [aGel,sNoGel]
# bGelVsNoGel = [bGel,vNoGel]
# hLGelVsNoGel = one_d_histogram(lGelVsNoGel, f'L gel vs no gel | {i}', min=1, hide=False, normalize=True)
# hAGelVsNoGel = one_d_histogram(aGelVsNoGel, f'A gel vs no gel | {i}', min=1, hide=False, normalize=True)
# hBGelVsNoGel = one_d_histogram(bGelVsNoGel, f'B gel vs no gel | {i}', min=1, hide=False, normalize=True)

# # CLAHE #
def clahe(img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = clahe.apply(img)
    return clahe

claheGelL = clahe(lGel)
claheGelA = clahe(aGel)
claheGelB = clahe(bGel)
claheNoGelL = clahe(lNoGel)
claheNoGelA = clahe(aNoGel)
claheNoGelB = clahe(bNoGel)

claheGelLAB = cv.merge((claheGelL, claheGelA, claheGelB))
claheNoGelLAB = cv.merge((claheNoGelL, claheNoGelA, claheNoGelB))

lGelVsNoGel = [claheGelL,claheNoGelL]
aGelVsNoGel = [claheGelA,claheNoGelA]
bGelVsNoGel = [claheGelB,claheNoGelB]
cv.imshow('LAB gel channels', np.hstack([claheGelL, claheGelA, claheGelB]))
cv.imshow('LAB no gel channels', np.hstack([claheNoGelL, claheNoGelA, claheNoGelB]))
cv.imshow('Originals', np.hstack([cv.bitwise_and(handGel, mHandGel), cv.bitwise_and(handNoGel, mHandNoGel)]))
hLGelVsNoGel = one_d_histogram(lGelVsNoGel, f'L gel vs no gel | {i}', min=5, hide=True, normalize=True)
hAGelVsNoGel = one_d_histogram(aGelVsNoGel, f'A gel vs no gel | {i}', min=5, hide=True, normalize=True)
hBGelVsNoGel = one_d_histogram(bGelVsNoGel, f'B gel vs no gel | {i}', min=5, hide=True, normalize=True)

plt.show()
cv.waitKey(0)