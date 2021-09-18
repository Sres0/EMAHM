import cv2 as cv
import numpy as np

# img = cv.imread('images/UV_img_sample.jpg')
img1 = cv.imread('images/Mano_lado.png')
img2 = cv.imread('images/Mano_centro.png')

# blank = np.zeros(img.shape[:2], dtype='uint8')
blank = np.zeros(img1.shape[:2], dtype='uint8')

def resize(img, name, w, h):
    cropped = cv.resize(img, (0,0), fx=w, fy=h)
    cv.imshow(name, cropped)

def split(blank, img, name):
    # h, w, ch = img.shape
    h = 0.5
    w = (h * 349)/345 #HOW DA HELL
    resize(img, name, w, h)
    # print(w, h) #698/459
    b,g,r = cv.split(img)
    # blue = cv.merge([b, blank, blank])
    # green = cv.merge([blank, g, blank])
    # red = cv.merge([blank, blank, r])

    resize(b, 'Blue', w, h) 
    resize(g, 'Green', w, h) 
    resize(r, 'Red', w, h) 
    # cv.imshow('blue', blue)
    # cv.imshow('green', green)
    # cv.imshow('red', red)

# merged = cv.merge([b,g,r])
# cv.imshow('Merged', merged)

# split(blank, img, 'UV')
split(blank, img1, 'Mano lado')
# split(blank, img2, 'Mano centro')

cv.waitKey(0)