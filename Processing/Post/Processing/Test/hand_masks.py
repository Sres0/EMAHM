import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

i = 44

path = 'Processing/Post/Bank/Imgs/hand_test_'+str(i)+'.png'

small_blur = (2, 2)
medium_blur = (5, 5)

img = cv.imread(path)

def createImg():
    w = int(1280/3)
    h = int(720/3)
    return cv.resize(img, (w,h))

img1=createImg()
cv.imshow('Original', img1)

def contour(mask, img, color, title, show=1):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    if show == 1:
        cv.imshow(title, img)

def blur(img, size, title, show=0):
    blurred= cv.blur(img, size)
    if show == 1:
        cv.imshow(title, blurred)
    return blurred

def threshold(img, min, max, title, method, show=1):
    _, thresh = cv.threshold(img, min, max, method)
    if show == 1:
        cv.imshow(title, thresh)
    return thresh

def threeDPlot(img, x, y, z, xlabel, ylabel, zlabel):
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(x.flatten(), y.flatten(), z.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_zlabel(zlabel)

def threeDHistogram(img, x, y, z):
    plt.figure()
    channels = (x, y, z)
    colors = ('b', 'g', 'r')
    for i, j in enumerate(channels):
        hist = cv.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, colors[i], label=j)
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.xlim([0,256])
    plt.legend()
    plt.grid()

def histogram(img):
    gray_hist = cv.calcHist([img], [0], None, [256], [0,256])

    plt.figure()
    plt.title('Grayscale Histogram')
    plt.xlabel('Bins') # Bins represent the intervals of pixel intensities--0 being black and 255, white
    plt.ylabel('# of pixels')
    plt.plot(gray_hist)
    plt.xlim([0,256])
    plt.grid()

##### HSV #####

# hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
# # cv.imshow('HSV', hsv)
# lower = np.array([0, 0, 100], dtype="uint8") # upper & lower pxl intensities for skin
# upper = np.array([150, 150, 255], dtype="uint8")

# mask = cv.inRange(hsv, lower, upper) # mask for normal img
# blurredMask = blur(mask, small_blur, 'Mascara difuminada', 1)
# hsvBlurred = cv.blur(hsv, small_blur) # Blur img with a 2x2 matrix
# blurredImgMask = cv.inRange(hsvBlurred, lower, upper) # mask for blurred img
# blurredImgMask = blur(blurredImgMask, small_blur, 'Mascara de imagen difuminada', 1)
# blurredImgNMask = blur(blurredImgMask, small_blur, 'Mascara difuminada de imagen difuminada', 1)

# img2 = createImg()
# img3 = createImg()

# contour(blurredMask, img1, (255, 255, 255), 'Contornos mascara difuminada')
# contour(blurredImgMask, img2, (255, 255, 255), 'Contornos mascara de imagen difuminada')
# contour(blurredImgNMask, img3, (255, 255, 255), 'Contornos mascara difuminada de imagen difuminada')

# cv.waitKey()

##### CHANNELS #####

# b,g,r = cv.split(img1)
# # cv.imshow('Verde', g)
# # cv.imshow('Rojo', r)
# # cv.imshow('Azul', r)

# gThresh = threshold(g, 65, 255, 'Verde 65', cv.THRESH_BINARY)
# rThresh = threshold(r, 100, 255, 'Rojo 100', cv.THRESH_BINARY)
# bThresh = threshold(b, 175, 255, 'Azul 175', cv.THRESH_BINARY)

# img2 = createImg()
# img3 = createImg()

# contour(gThresh, img1, (0, 255, 0), 'Verde')
# contour(rThresh, img2, (0, 0, 255), 'Rojo')
# contour(bThresh, img3, (255, 0, 0), 'Azul')

# img4 = createImg()
# img5 = createImg()

# gBlurredThresh = cv.blur(gThresh, (2,2))
# gBlurredThresh = threshold(gBlurredThresh, 65, 255, 'Verde; mascara difuminada 65', cv.THRESH_BINARY)

# gBlurredImg = cv.blur(g, small_blur)
# gBlurredImgThresh = threshold(gBlurredImg, 65, 255, 'Verde; mascara imagen difuminada 65', cv.THRESH_BINARY)

# contour(gBlurredThresh, img5, (255, 255, 255), 'Verde; contorno mascara difuminada')
# contour(gBlurredImgThresh, img4, (255, 255, 255), 'Verde; contorno mascara imagen difuminada')

# cv.waitKey()

##### BRIGHTNESS & CONTRAST #####

# # cv.imshow('Original', img1)
# a = 1 # Contrast
# # a = 1 # Contrast
# b = -100 # Brightness
# lower = np.array([255, 95, 160], dtype='uint8') # Contrast 3
# upper = np.array([255, 255, 255], dtype='uint8')
# # lower = np.array([0, 0, 0], dtype='uint8') # Contrast 1. Inverso
# # upper = np.array([100, 255, 150], dtype='uint8') # Contrast 1. Inverso

# leveled = cv.convertScaleAbs(img1, alpha=a, beta=b)
# cv.imshow('Brillo y contraste', leveled)

# leveledBlurred = cv.blur(leveled, small_blur)
# blurredImgMask = cv.inRange(leveled, lower, upper)
# # blurredImgMask = blur(cv.bitwise_not(blurredImgMask), small_blur, 'Mascara imagen difuminada', 1) # inverso
# blurredImgMask = blur(blurredImgMask, small_blur, 'Mascara imagen difuminada', 1)

# mask = cv.inRange(leveled, lower, upper) # Canal verde
# cv.imshow('Mascara', mask)
# # blurredMask = blur(cv.bitwise_not(mask), medium_blur, 'Mascara difuminada', 1) # inverso
# blurredMask = blur(mask, medium_blur, 'Mascara difuminada', 1)

# img2 = createImg()
# img3 = createImg()

# contour(mask, img1, (255, 255, 255), 'Contorno mascara')
# contour(blurredImgMask, img2, (255, 255, 255), 'Contorno mascara imagen difuminada')
# contour(blurredMask, img3, (255, 255, 255), 'Contorno mascara difuminada')

# cv.waitKey()

##### COMBINATION #####

# hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# hsv = blur(hsv, small_blur, 'HSV', 1)
# cv.imshow('Original', img1)
# cv.imshow('HSV', hsv)
# cv.imshow('Grayscale', gray)

# Plot #
# b, g, r = cv.split(img1)
# h, s, v = cv.split(hsv)

# Histogram #

# threeDPlot(img1, b, g, r, 'Blue', 'Green', 'Red')
# threeDPlot(hsv, h, s, v, 'Hue', 'Saturation', 'Value')
# threeDHistogram(img1, 'blue', 'green', 'red')
# threeDHistogram(hsv, 'hue', 'saturation', 'value')
# histogram(gray)

# Mask #

img2 = createImg()
img3 = createImg()
img4 = createImg()

maskGray = blur(gray, medium_blur, 'blurred gray')

# lower = np.array([115, 150, 0], dtype='uint8') # 112, 145, 25
# lower = np.array([112, 145, 25], dtype='uint8') # 112, 145, 25
# upper = np.array([140, 198, 255], dtype='uint8') # 142, 200, 200

# maskHSV = cv.inRange(hsv, lower, upper)
maskGray = threshold(maskGray, 110, 255, 'Mascara gris binary', cv.THRESH_BINARY_INV, 0) # 105, 240
# maskGray = threshold(gray, 110, 255, 'Mascara gris otsu', cv.THRESH_OTSU) # 105, 240
# cv.imshow('Mascara HSV', maskHSV)
# cv.imshow('Mascara gris', maskGray)

# contour(maskHSV, img1, (255, 255, 255), 'Contorno mascara HSV')
# contour(maskGray, img2, (255, 255, 255), 'Contorno mascara gris')

# Morphology #

def morph(opt, src, title, kernel, show=0, iterations=1):
    
    if opt == 'e':
        mask = cv.erode(src, kernel, iterations=iterations)
    elif opt == 'd':
        mask = cv.dilate(src, kernel, iterations=iterations)
    elif opt == 'o':
        mask = cv.morphologyEx(src, cv.MORPH_OPEN, kernel)
    else:
        mask = cv.morphologyEx(src, cv.MORPH_CLOSE, kernel)
    if show == 1:
        cv.imshow(title, mask)
        contour(mask, img1, (255, 255, 255), 'Contorno '+title)
    return mask

small_kernel = np.ones((2,2),np.uint8)
smedium_kernel = np.ones((3,3),np.uint8)
medium_kernel = np.ones((5,5),np.uint8)

# maskedGray = morph('d', maskGray, 'mascara gris dilatada', medium_kernel, 0)
# maskedGray = morph('e', maskedGray, 'mascara gris erosionada', medium_kernel, 0)
# maskedGray = morph('o', maskedGray, 'mascara gris abierta', small_kernel, 0)
maskedGray = morph('c', maskGray, 'mascara gris cerrada', medium_kernel, 0)
# maskGrayDilate = morph('d', maskGrayErode, 'mascara gris dilatada', small_kernel, 1, iterations=2)

# maskedGray = blur(maskGray, (4,4), 'erode', 1)
# maskGrayDilate = morph('d', maskedGray, 'mascara gris dilatada', small_kernel, 0, iterations=100)
# maskedGray = 255 - maskedGray # invert

# maskedGray = threshold(maskedGray, 110, 255, 'Mascara gris thresh', cv.THRESH_BINARY, 1)

cropped = cv.bitwise_and(img1, img1, mask=cv.bitwise_not(maskedGray))

# GEL #

# gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
# lab = cv.cvtColor(cropped, cv.COLOR_BGR2LAB)
# b, g, r = cv.split(cropped)
h, s, v = cv.split(hsv)
# l, a, b_lab = cv.split(lab) # Another option is combining channels

# cv.imshow('Recortada', cropped)
# cv.imshow('HSV', hsv)
# cv.imshow('Gris', gray)
# cv.imshow('LAB', lab)

# cv.imshow('light', l)
# cv.imshow('A', a)
# cv.imshow('B', b_lab)

# Histograms #

# threeDHistogram(cropped, 'blue', 'green', 'red')
# threeDHistogram(hsv, 'hue', 'saturation', 'value')
# threeDHistogram(lab, 'Lightness', 'A color range', 'B color range')
# histogram(gray)

# Masks #

# greenMask = threshold(g, 190, 255, 'mascara verde', cv.THRESH_BINARY, 0)
# blueMask = threshold(b, 245, 255, 'mascara azul', cv.THRESH_BINARY, 0)
# redMask = threshold(r, 215, 255, 'mascara roja', cv.THRESH_BINARY, 0)
# grayMask = threshold(gray, 200, 255, 'mascara gris', cv.THRESH_BINARY, 0)
# aMask = threshold(a, 145, 255, 'mascara brillo', cv.THRESH_BINARY_INV, 0)
hMask = threshold(h, 128, 255, 'mascara brillo', cv.THRESH_BINARY_INV, 0)
# cv.imshow('hue mask', hMask)
hMask = cv.bitwise_and(hMask, hMask, mask=cv.bitwise_not(maskedGray))
# cv.imshow('hue mask 2', hMask)

# Contours #

img5 = createImg()
contour(maskedGray, img1, (255, 255, 255), 'mascara', 0)
contour(maskedGray, img2, (255, 255, 255), 'mascara', 0)
contour(maskedGray, img3, (255, 255, 255), 'mascara', 0)
contour(maskedGray, img5, (255, 255, 255), 'mascara', 0)
# contour(greenMask, img1, (0, 255, 0), 'contorno mascara verde')
# contour(grayMask, img2, (255, 255, 255), 'contorno mascara gris')
# contour(blueMask, img3, (255, 0, 0), 'contorno mascara azul')
# contour(redMask, img3, (0, 0, 255), 'contorno mascara roja')
# contour(aMask, img3, (255, 255, 255), 'contorno mascara brillo')
# contour(hMask, img4, (255, 255, 255), 'contorno mascara hue')

# Morphology #

maskedGel = blur(hMask, small_blur, 'blur gel', 1)
# maskedGel = morph('e', hMask, 'mascara gel erosionada', small_kernel, 0)
# maskedGel = morph('d', maskedGel, 'mascara gel dilatada', medium_kernel, 0)
maskedGel = morph('c', maskedGel, 'mascara gel cerrada', small_kernel, 0)
maskedGel = morph('o', maskedGel, 'mascara gel abierta', smedium_kernel, 0)

# maskedGel = threshold(maskedGel, 70, 255, 'mascara gel', cv.THRESH_BINARY, 1)
maskedGel = cv.bitwise_and(maskedGel, maskedGel, mask=cv.bitwise_not(maskedGray))

# cv.imshow('mascara gel', maskedGel)
contour(maskedGel, img2, (255, 255, 255), 'contorno mascara gel') # Color piel dada


plt.show()

cv.waitKey()