import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

i = 510

def path(i):
    return f'Processing/Bank/Imgs/hand_test_{i}.png'

def createImg(i, hide=True, resized=False):
    img = cv.imread(path(i))
    if not resized:
        w = int(1280/3)
        h = int(720/3)
        img = cv.resize(img, (w,h))
    if not hide: cv.imshow(f'Original {i}', resized)
    return img

img = createImg(i, hide=True, resized=True)
img_blur = cv.blur(img, (2,2))

edges = cv.Canny(img_blur, 130, 255)
# Display Canny Edge Detection Image
cv.imshow('Canny Edge Detection', edges)
cv.waitKey(0)

cv.waitKey(0) & 0xFF=='q'
cv.destroyAllWindows()