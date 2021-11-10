import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

i = 27
j = 29

def path(i):
    return 'Processing/Post/Bank/Imgs/hand_test_'+str(i)+'.png'

smallBlur = (2, 2)
mediumBlur = (5, 5)

img1 = cv.imread(path(i))
img2 = cv.imread(path(j))
w = int(1280/3)
h = int(720/3)
img1 = cv.resize(img1, (w,h))
img2 = cv.resize(img2, (w,h))
cv.imshow('original antes', img1)
cv.imshow('original despues', img2)

def contour(mask, img, color, title):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    cv.imshow(title, img)

def blur(img, size, title):
    blurred= cv.blur(img, size)
    # cv.imshow(title, blurred)
    return blurred

def threshold(channel, min, max, title):
    _, thresh = cv.threshold(channel, min, max, cv.THRESH_BINARY)
    # cv.imshow(title, thresh)
    return thresh

b1,g1,r1 = cv.split(img1)
b2,g2,r2 = cv.split(img2)
cv.imshow('verde', g2)

gThresh1 = threshold(g1, 75, 255, f'Verde {i}')
gThresh1 = blur(gThresh1, (2,2), f'Verde {i}')
gThresh2 = threshold(g2, 65, 255, f'Verde {j}')

_, ggel_thresh2 = cv.threshold(g2, 150, 255, cv.THRESH_BINARY)
ggel_thresh2 = cv.bitwise_and(gThresh2, ggel_thresh2)

contour(gThresh1, img1, (0, 255, 0), str(i))    
contour(ggel_thresh2, img2, (0, 255, 0), str(j))

hand = cv.countNonZero(gThresh1)
gel = cv.countNonZero(ggel_thresh2)
print(100 - (((hand-gel)/hand) * 100), '%')
# percentage = (hand-gel)/hand * 100
# print(f'{percentage} %')

# ARUCO

cv.waitKey()

# i = '2C358F95-4BED-4860-8177-45B86FFBF3BA'
# img_path = 'Processing/Post/Bank/Imgs/'+str(i)+'.jpg'
# # img_path = 'Processing/Post/Bank/Imgs/hand_test_'+str(i)+'.png'
# img = cv.imread(img_path)

# # define the upper and lower boundaries of the HSV pixel intensities 
# # to be considered 'skin'
# hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# lower = np.array([0, 48, 80], dtype="uint8")
# upper = np.array([20, 255, 255], dtype="uint8")
# skinMask= cv.inRange(hsvim, lower, upper)
# # 
# # blur the mask to help remove noise
# skinMask= cv.blur(skinMask, (2, 2))
# cv.imshow('skinMask', skinMask)

# # # get threshold image
# ret, thresh = cv.threshold(skinMask, 100, 255, cv.THRESH_BINARY)
# cv.imshow("thresh", thresh)

# # draw the contours on the empty image
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, -1, (255, 255, 255), 1)
# cv.imshow("contours", img)

# cv.waitKey()