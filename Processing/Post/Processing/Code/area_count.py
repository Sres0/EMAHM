import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

i = 21
j = 0

def path(i):
    return f'Processing/Post/Bank/Imgs/hand_test_{i}.png'

# img = cv.imread('Processing/Post/Bank/Imgs/hand_test_0.png')
def create_img(i, aux):
    if aux == 0: img = cv.imread(f'Processing/Post/Processing/Registry/6. Levels/0.{i}. bgr_{i}.png')
    else: img = cv.imread('Processing/Post/Bank/Imgs/hand_test_'+str(i)+'.png')
    # img = cv.imread(path(i))
    w = int(1280/3)
    h = int(720/3)
    return cv.resize(img, (w,h))


def contour(img, thresh, color, i, j, name):
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    # cv.imwrite('Processing/Post/Processing/Registry/5. Area Count/0.'+str(i)+'. area_count_'+str(j)+'_.png', g)
    cv.imshow(name, img)
    # return 'Processing/Post/Processing/Registry/5. Area Count/0.'+str(i)+'. area_count_'+str(j)+'_.png'

# def plot(path, j, title):
#     pltimg = mpimg.imread(path)
#     plt.subplot(4,4,j+1)
#     plt.axis('off')
#     plt.title(title)
#     plt.imshow(pltimg, cmap='gray')
#     return j+1

# plt.figure()

# for i in range(4):
img = create_img(i, 1)
img2 = create_img(i, 1)
img3 = create_img(i, 1)
cv.imshow('Original', img)
# j = plot(path(i), j, 'Original')
# print(path(i))

b,g,r = cv.split(img)
# cv.imshow('Verde', g)

ret, hand_thresh = cv.threshold(g, 75, 255, cv.THRESH_BINARY)
# cv.imshow('Mascara mano', hand_thresh)
ret, ggel_thresh = cv.threshold(g, 180, 255, cv.THRESH_BINARY)
# cv.imshow('Mascara verde gel', ggel_thresh)
ret, rgel_thresh = cv.threshold(r, 195, 255, cv.THRESH_BINARY)
# cv.imshow('Mascara rojo gel', rgel_thresh)
ret, bgel_thresh = cv.threshold(b, 245, 255, cv.THRESH_BINARY)
# cv.imshow('Mascara azul gel', bgel_thresh)

# hand = create_img(i, 1)
# bHand,gHand,rHand = cv.split(hand)
# ret, hand_thresh = cv.threshold(gHand, 75, 255, cv.THRESH_BINARY)
# cv.imshow('Mascara mano', hand_thresh)
# g_mask = cv.bitwise_and(gHand, gHand, mask=hand_thresh)
# cv.imshow('Mascara verde', g_mask)
# ret, ggel_thresh = cv.threshold(g_mask, 75, 255, cv.THRESH_BINARY)
# cv.imshow('Gel verde', ggel_thresh)
# ggel_thresh = cv.bitwise_not(ggel_thresh)
# ret, rgel_thresh = cv.threshold(r, 10, 255, cv.THRESH_BINARY)
# cv.imshow('Mascara rojo gel', rgel_thresh)
# ret, bgel_thresh = cv.threshold(b, 10, 255, cv.THRESH_BINARY)
# cv.imshow('Mascara azul gel', bgel_thresh)

contour(img, ggel_thresh, (0,255,0), i, j, 'Verde 180')
# j = plot(path(i), j, 'Verde 180')
contour(img2, rgel_thresh, (0,0,255), i, j, 'Rojo 195')
# j = plot(path(i), j, 'Rojo 195')
contour(img3, bgel_thresh, (255,0,0), i, j, 'Azul 245')
# j = plot(path(i), j, 'Azul 245')

h_count = cv.countNonZero(hand_thresh)
g_count = cv.countNonZero(ggel_thresh)
print('Mano:', h_count)
print('Gel:', g_count)
print('Porcentaje sucio:', round(g_count/h_count * 100, 2), '%')

# plt.show()
cv.waitKey(0)