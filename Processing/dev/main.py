import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

results = []
i = 634 # con gel
j = i - 2

def path(i): return f'Processing/Bank/Imgs/hand_test_{i}.png'

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

def one_d_histogram(img, title, hide=True, min=1, normalize=False):
    hist = []
    
    if not hide: set_histogram(title)
    if type(img) == list:
        for cnl in img:
            h = cv.calcHist([cnl], [0], None, [255], [min, 256])
            if normalize: cv.normalize(h,h,0,1,cv.NORM_MINMAX)
            if not hide: plt.plot(h)
            hist.append([y[0] for y in h])
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

mGrayBlurredHandGel = get_binary_mask(i, find_hand_thresh(hGrayBlurredHandGel, title='gel thresh', hide=True), grayBlurredHandGel, 'gray blurred gel')
mGrayBlurredHandNoGel = get_binary_mask(i, find_hand_thresh(hGrayBlurredHandNoGel, title='no gel thresh', hide=True), grayBlurredHandNoGel, 'gray blurred no gel')

def contour(mask, img, color, i, title, hide=True):
    blank = np.zeros((img.shape), dtype=np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 3000]
    for cnt in contours: 
        cv.drawContours(blank, [cnt], -1, color, -1)
        cv.drawContours(img, [cnt], -1, color, 1)
    if not hide:
        cv.imshow(title+f' {i}', np.hstack([img,handNoGel]))
        results.append(img)
    areas = [cv.contourArea(cnt) for cnt in contours]
    return blank, sum(areas)

copyHandGel = handGel.copy()
copyHandNoGel = handNoGel.copy()
mHandGel, aHandGel = contour(mGrayBlurredHandGel, copyHandGel, (255, 255, 255), i, 'gray blurred gel', hide=True)
mHandNoGel, aHandNoGel = contour(mGrayBlurredHandNoGel, copyHandNoGel, (255, 255, 255), j, 'gray blurred no gel', hide=True)

### GEL ###

# HSV #
def split_n_clahe(img):
    x,y,z = cv.split(img)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    for chan in (x,y,z):
        chan = clahe.apply(chan)
    img = cv.merge((x,y,z))
    return x,y,z,img

imgHsvGel = cv.cvtColor(cv.bitwise_and(handGel, mHandGel), cv.COLOR_BGR2HSV)
imgHsvNoGel = cv.cvtColor(cv.bitwise_and(handNoGel, mHandNoGel), cv.COLOR_BGR2HSV)

claheGelH, claheGelS, claheGelV, claheGelHSV = split_n_clahe(imgHsvGel)
claheNoGelH, claheNoGelS, claheNoGelV, claheNoGelHSV = split_n_clahe(imgHsvNoGel)

hGelVsNoGel = [claheGelH,claheNoGelH]
hHGelVsNoGel = one_d_histogram(hGelVsNoGel, f'Hue gel vs no gel | {i}', min=1, hide=True, normalize=True)

imgHsvgel = cv.merge((claheGelH, claheGelS, claheGelV))
imgHsvNoGel = cv.merge((claheNoGelH, claheNoGelS, claheNoGelV))

def find_gel_percentage(hHue, hide=True, title='Presencia de gel'):
    hThreshMax = 254
    diff = []
    x0 = []
    # Hue
    for x, y1 in enumerate(hHue[0]):
        diff.append(np.sign(y1-hHue[1][x]))
        if y1 > 0.1: x0.append(x)
    for arg in np.argwhere(np.diff(diff)).flatten():
        if arg in x0: hThreshMax = arg + 1
    lower = np.array([10, 10, 10], dtype='uint8')
    upper = np.array([hThreshMax, 255, 255], dtype='uint8')
    if hThreshMax != 254:
        copyHandGel = handGel.copy()
        mGel = cv.inRange(imgHsvGel, lower, upper)
        mGel, aGel = contour(mGel, copyHandGel, (255, 255, 255), i, title, hide=False)
        percentage = round((aGel*100)/aHandGel,2)
        print('Porcentaje lavado:',percentage,'%')
    else:
        percentage=0
        print('Error: porcentaje de gel no identificado.')
    if not hide: 
        set_histogram(title)
        x = np.array(range(0,255))
        plt.plot(x,hHue[0])
        plt.plot(x,hHue[1])
        plt.plot(hThreshMax,hHue[0][hThreshMax], "x")
    return percentage

gelPercentage = find_gel_percentage(hHGelVsNoGel, hide=True)

plt.show()
cv.waitKey(0)