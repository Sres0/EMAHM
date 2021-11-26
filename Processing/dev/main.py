import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
# from save import res

i = 622 # con gel
j = i - 2

def path(i):
    return f'Processing/Bank/Imgs/hand_test_{i}.png'

def createImg(i, hide=True, resized=False):
    img = cv.imread(path(i))
    if not resized:
        w = int(1280/3)
        h = int(720/3)
        img = cv.resize(img, (w,h))
        img = img[0:int((720/3)*4/5), 0:int(1280/3)] # recortar imagen en muñeca
    if not hide: cv.imshow(f'Original {i}', img)
    return img

handGel = createImg(i, hide=False)
handNoGel = createImg(j, hide=False)
# cv.imshow('Segmented gel & no gel', np.hstack([HandGel, HandNoGel]))

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

def one_d_histogram(img, title, hide=True):
    if not hide: set_histogram(title)
    
    if type(img) == list:
        for cnl in img:
            hist = cv.calcHist([cnl], [0], None, [255], [0, 256])
            if not hide: plt.plot(hist)
    else:
        hist = cv.calcHist([img], [0], None, [255], [0, 256])
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
    if type(thresh) == list:
        pass
    else:
        _, mask = cv.threshold(img, thresh, 255, method)
    if not hide: cv.imshow(f'Mascara {channel_name} {i} | {thresh}', mask)
    return mask

mGrayBlurredHandGel = get_binary_mask(i, find_hand_thresh(hGrayBlurredHandGel, title='gel thresh', hide=True), grayBlurredHandGel, 'gray blurred gel', hide=True)
mGrayBlurredHandNoGel = get_binary_mask(i, find_hand_thresh(hGrayBlurredHandNoGel, title='no gel thresh', hide=True), grayBlurredHandNoGel, 'gray blurred no gel')
# cv.imshow('Segmented gel & no gel mask', np.hstack([mGrayBlurredHandGel, mGrayBlurredHandNoGel]))

### PENDING MORPH

def contour(mask, img, color, i, title, hide=True):
    blank = np.zeros((img.shape), dtype=np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 1000]
    for cnt in contours:
        cv.drawContours(img, [cnt], -1, color, 1)
        cv.drawContours(blank, [cnt], -1, color, -1)
    # if not hide: cv.imshow(f'contorno {title} {i} y mascara', np.hstack([img, blank]))
    if not hide: cv.imshow(title+f' {i}', np.hstack([img, blank]))
    return blank, [cv.contourArea(cnt) for cnt in contours]

copyHandGel = handGel.copy()
copyHandNoGel = handNoGel.copy()
mHandGel, aHandGel = contour(mGrayBlurredHandGel, copyHandGel, (255, 255, 255), i, 'gray blurred gel', hide=True)
mHandNoGel, aHandNoGel = contour(mGrayBlurredHandNoGel, copyHandNoGel, (255, 255, 255), j, 'gray blurred no gel', hide=True)
# cv.imshow('Segmented gel & no gel contour', np.hstack([copyHandGel, copyHandNoGel]))

### Gel ###

# imgLabGel = cv.cvtColor(handGel, cv.COLOR_BGR2LAB)
imgYCrCbGel = cv.cvtColor(cv.bitwise_and(handGel, mHandGel), cv.COLOR_BGR2YCrCb)
yGel,crGel,cbGel = cv.split(imgYCrCbGel)
bgrGel = [yGel,crGel,cbGel]
# imgHsvNoGel = cv.cvtColor(cv.bitwise_and(handNoGel, mHandNoGel), cv.COLOR_BGR2HSV)
# hNoGel,sNoGel,vNoGel = cv.split(cv.cvtColor(imgHsvNoGel, cv.COLOR_BGR2HSV))
# hsvNoGel = [hNoGel,sNoGel,vNoGel]
# cv.imshow('hsv', np.hstack([imgHsvGel, imgHsvNoGel]))

# copyHandGel2 = handGel.copy()
# copyHandNoGel2 = handNoGel.copy()
# hHsvGel = one_d_histogram(hsvGel, f'HSV Gel {i}', hide=False)
# hHsvNoGel = one_d_histogram(hsvNoGel, f'HSV no Gel {i}', hide=False)
# hHGel = one_d_histogram(hsvGel[0], f'Hue Gel {i}', hide=True)
# sHGel = one_d_histogram(hsvGel[1], f'Sat Gel {i}', hide=True)
# vHGel = one_d_histogram(hsvGel[2], f'Val Gel {i}', hide=True)
# hNoHGel = one_d_histogram(hsvNoGel[0], f'Hue no Gel {i}', hide=True)
# sNoHGel = one_d_histogram(hsvNoGel[1], f'Sat no Gel {i}', hide=True)
# vNoHGel = one_d_histogram(hsvNoGel[2], f'Val no Gel {i}', hide=True)

def nothing(x):
    pass

cv.namedWindow(f'Gel {i}')
cv.createTrackbar('Y min', f'Gel {i}', 0, 255, nothing)
cv.createTrackbar('Y max', f'Gel {i}', 255, 255, nothing)
cv.createTrackbar('Cr min', f'Gel {i}', 0, 255, nothing)
cv.createTrackbar('Cr max', f'Gel {i}', 255, 255, nothing)
cv.createTrackbar('Cb min', f'Gel {i}', 0, 255, nothing)
cv.createTrackbar('Cb max', f'Gel {i}', 255, 255, nothing)

b_min = 0
g_min = 0
r_min = 0
b_max = 255
g_max = 255
r_max = 255

while True:
    img = handGel.copy()
    lower = np.array([b_min, g_min, r_min], dtype='uint8')
    upper = np.array([b_max, g_max, r_max], dtype='uint8')
    mGel = cv.inRange(imgYCrCbGel, lower, upper)
    # mGel = get_binary_mask(i, thresh, img, 'green')
    mGel, aGel = contour(mGel, img, (255, 255, 255), i, f'Gel {i}', hide=True)
    cv.imshow(f'Gel {i}', img)
    if cv.waitKey(0) & 0xFF == 'q':
        break
    b_min = cv.getTrackbarPos('Y min', f'Gel {i}')
    b_max = cv.getTrackbarPos('Y max', f'Gel {i}')
    g_min = cv.getTrackbarPos('Cr min', f'Gel {i}')
    g_max = cv.getTrackbarPos('Cr max', f'Gel {i}')
    r_min = cv.getTrackbarPos('Cb min', f'Gel {i}')
    r_max = cv.getTrackbarPos('Cb max', f'Gel {i}')

plt.show()
cv.waitKey(0)