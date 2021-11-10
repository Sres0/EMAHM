import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

i = 0

# for i in range(8):
img = cv.imread('Processing/Post/Bank/Imgs/hand_test_'+str(i)+'.png')
w = int(1280/3)
h = int(720/3)
img = cv.resize(img, (w,h))

### DIDNT WORK ###
# a = 1 # Contrast (1.0 - 3.0)
# b = 50 # Brightness (0 - 100)

# leveled = cv.convertScaleAbs(img, alpha=a, beta=b)
# cv.imshow('Original', img)
# cv.imshow('Modified', leveled)

imghsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# imghsv[:,:,2] = [[max(px - 25, 0) if px < 230 else min(px + 25, 255) for px in row] for row in imghsv[:,:,0]] # Hue
imghsv[:,:,2] = [[max(px - 25, 0) if px < 25 else min(px + 25, 255) for px in row] for row in imghsv[:,:,1]] # Sat
# imghsv[:,:,2] = [[max(px - 25, 0) if px < 25 else min(px + 25, 255) for px in row] for row in imghsv[:,:,2]] # Val
cv.imshow('Original', img)
cv.imshow('HSV', imghsv)
bgr = cv.cvtColor(imghsv, cv.COLOR_HSV2BGR)
cv.imshow('BGR', bgr)
bnw = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
cv.imshow('B&W', bnw)
    # cv.imwrite(f'Processing/Post/Processing/Registry/6. Levels/0.{i}. bgr_{i}.png', bgr)

cv.waitKey(0)