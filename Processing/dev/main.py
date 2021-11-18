import cv2 as cv
import matplotlib.pyplot as plt

i = 44 # con gel
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

imgGel = createImg(i, 1)

### BLUR ###

medium_blur = (3,3)

blurredImgGel = cv.blur(imgGel, medium_blur)
grayBlurredImgGel = cv.blur(cv.cvtColor(blurredImgGel, cv.COLOR_BGR2GRAY), medium_blur)

def one_d_histogram(img, title):
    lColors = ('b', 'g', 'r')
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('Pixeles')
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

def threshold(i, mval, img, channel_name, method=cv.THRESH_BINARY, hide=False):
    _, thresh = cv.threshold(img, mval, 255, method)
    if not hide: cv.imshow(f'Mascara {channel_name} {i} | {mval}', thresh)
    return thresh

mGrayBlurredImgGel = threshold(i, 87.7, grayBlurredImgGel, 'gray blurred', hide=True)
mGrayBlurredImgGel = cv.bitwise_and(imgGel, imgGel, mask=mGrayBlurredImgGel)
cv.imshow('mask gray blurred gel', mGrayBlurredImgGel)

# def cvt_n_split(img, method=cv.COLOR_BGR2HSV):
#     img = cv.cvtColor(img, method)
#     x, y, z = cv.split(img)
#     return [x, y, z, img]


plt.show()
cv.waitKey(0)