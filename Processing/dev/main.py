import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
# from save import res

i = 618 # con gel
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

### Backprojection # How'd the machine know the roi
# def backproject(roiHist, hsv, img):
#     cv.normalize(roiHist,roiHist,0,1,cv.NORM_MINMAX)
#     dst = cv.calcBackProject([hsv],[0,1],roiHist,[0,180,0,256],1)
#     disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
#     cv.filter2D(dst,-1,disc,dst)
#     _,thresh = cv.threshold(dst,50,255,0)
#     thresh = cv.merge((thresh,thresh,thresh))
#     res = cv.bitwise_and(img,thresh)
#     cv.imshow('backprojection', np.vstack((img,thresh,res)))

def one_d_histogram(img, title, hide=True, min=0, bProject=False):
    if not hide: set_histogram(title)
    
    if type(img) == list:
        for cnl in img:
            hist = cv.calcHist([cnl], [0], None, [255], [min, 256])
            cv.normalize(hist,hist,0,1,cv.NORM_MINMAX)
            if not hide: plt.plot(hist)
    else:
        hist = cv.calcHist([img], [0], None, [255], [min, 256])
        cv.normalize(hist,hist,0,1,cv.NORM_MINMAX)
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
imgLabGel = cv.cvtColor(cv.bitwise_and(handGel, mHandGel), cv.COLOR_BGR2LAB)
lGel,aGel,bGel = cv.split(imgLabGel)
imgLabNoGel = cv.cvtColor(cv.bitwise_and(handNoGel, mHandNoGel), cv.COLOR_BGR2HSV)
hNoGel,sNoGel,vNoGel = cv.split(imgLabNoGel)
# cv.imshow('hsv', np.hstack([imgHsvGel, imgHsvNoGel]))

lGelVsNoGel = [lGel,hNoGel]
aGelVsNoGel = [aGel,sNoGel]
bGelVsNoGel = [bGel,vNoGel]
hLGelVsNoGel = one_d_histogram(lGelVsNoGel, f'L gel vs no gel | {i}', min=1, hide=False)
hAGelVsNoGel = one_d_histogram(aGelVsNoGel, f'A gel vs no gel | {i}', min=1, hide=False)
hBGelVsNoGel = one_d_histogram(bGelVsNoGel, f'B gel vs no gel | {i}', min=1, hide=False)

### Grabcut

# mHandGel[mHandGel > 0] = cv.GC_PR_FGD
# mHandGel[mHandGel == 0] = cv.GC_BGD
# fgModel = np.zeros((1, 65), dtype="float")
# bgModel = np.zeros((1, 65), dtype="float")
# # (mask, bgModel, fgModel) = cv.grabCut(imgHsvGel, mHandGel, None, bgModel, fgModel, mode=cv.GC_INIT_WITH_MASK, iterCount=6)
# # gCuts = (("Definite Background", cv.GC_BGD),
# # 	("Probable Background", cv.GC_PR_BGD),
# # 	("Definite Foreground", cv.GC_FGD),
# # 	("Probable Foreground", cv.GC_PR_FGD))

# # for (name, val) in gCuts: valueMask = (mask == val).astype("uint8") * 255

# # outputMask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1)
# # outputMask = (outputMask * 255).astype("uint8")
# # output = cv.bitwise_and(handGel, handGel, mask=outputMask)

# # cv.imshow("Input", handGel)
# # cv.imshow("GrabCut Mask", outputMask)
# # cv.imshow("GrabCut Output", output)

plt.show()
cv.waitKey(0)