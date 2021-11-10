import cv2 as cv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

files = []

def save(img, i, name):
    cv.imshow(name, img)
    file = f'Processing/Post/Processing/Registry/3. Green/3.{i}. hand_test_4_{name}_green.png'
    files.append(file)
    cv.imwrite(file, img)

img = cv.imread('Processing/Post/Bank/Imgs/hand_test_4.png')
w = int(1280/3)
h = int(720/3)
img = cv.resize(img, (w,h))
# cv.imshow('Original', img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
b,g,r = cv.split(img)
cv.imwrite('Processing/Post/Processing/Registry/3. Green/0.0. hand_test_4_green.png', g)

### LAPLACIAN ###
lap = cv.Laplacian(g, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
save(lap, 0, 'laplacian')

### SOBEL ###
sobelx = cv.Sobel(g, cv.CV_64F, 1, 0)
sobely = cv.Sobel(g, cv.CV_64F, 0, 1)
combined = cv.bitwise_or(sobelx, sobely)

save(sobelx, 1, 'sobel X')
save(sobely, 2, 'sobel Y')
save(combined, 3, 'combined')

### CANNY ###
canny = cv.Canny(g, 40, 90)
save(canny, 4, 'canny')

### THRESH ###
ret, thresh = cv.threshold(g, 80, 255, cv.THRESH_BINARY)
save(thresh, 5, 'thresh')
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # The most important
# print(f'{len(contours)} contours found')
blank = np.zeros(img.shape, dtype='uint8')
cv.drawContours(blank, contours, -1, (255,255,255), 1)
save(blank, 6, 'contours')

def plot(name, i, title, vmin, vmax):
    pltimg = mpimg.imread(name)
    plt.subplot(3,3,i)
    plt.axis('off')
    plt.title(title)
    if vmin == False:
        plt.imshow(pltimg, cmap='gray')
    else:
        plt.imshow(pltimg, cmap='gray', vmin=vmin, vmax=vmax)

plt.figure()
plot('Processing/Post/Bank/Imgs/hand_test_4.png', 1, 'Original', False, False)
plot('Processing/Post/Processing/Registry/3. Green/0.0. hand_test_4_green.png', 2, 'green channel', False, False)
i = 3

for file in files:
    title = file.replace('Processing/Post/Processing/Registry/3. Green/3.', '')
    title = title.replace('hand_test_4_', '')
    title = title.replace('_green.png', '')
    if i in range(4,7):
        vmin = 0
        vmax = 0.025
    else:
        vmin = 0
        vmax = 0.8
    plot(file, i, title, vmin, vmax)
    i += 1

plt.show()

cv.waitKey(0)