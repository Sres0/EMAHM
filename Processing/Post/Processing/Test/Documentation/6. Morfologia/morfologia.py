import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

i = 76 # con gel
j = 64

def path(i):
    return f'Processing/Post/Bank/Imgs/hand_test_{i}.png'

def createImg(path, i, hide=False, write=False):
    img = cv.imread(path)
    w = int(1280/3)
    h = int(720/3)
    resized = cv.resize(img, (w,h))
    if not hide: cv.imshow(f'Original {i}', resized)
    if write: cv.imwrite(f'Processing/Post/Processing/Test/Documentation/6. Morfologia/original_{i}.png', resized)
    return resized

img = createImg(path(i), i, write=True)
img2 = createImg(path(j), j, write=True)
hsvEdited = createImg(path(i), i, 1)
hsvEditedBlur = cv.blur(hsvEdited, (2,2))

### Máscara y morf ###

def threshold(i, mval, img, channel_name, method=cv.THRESH_BINARY, write=False):
    _, thresh = cv.threshold(img, mval, 255, method)
    cv.imshow(f'Mascara {channel_name} {i} | {mval}', thresh)
    if write: cv.imwrite(f'Processing/Post/Processing/Test/Documentation/6. Morfologia/thresh_{channel_name}_{i}.png', thresh)
    return thresh

def contour(mask, img, color, i, channel_name, write=False):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, color, 1)
    cv.imshow(f'contorno {channel_name} {i}', img)
    if write: cv.imwrite(f'Processing/Post/Processing/Test/Documentation/6. Morfologia/contour_{channel_name}_{i}.png', img)

def morph(opt, src, img, title, kernel, hide=False, iterations=1, write=False):
    mask = 0
    if opt == 'e':
        mask = cv.erode(src, kernel, iterations=iterations)
    elif opt == 'd':
        mask = cv.dilate(src, kernel, iterations=iterations)
    elif opt == 'o':
        mask = cv.morphologyEx(src, cv.MORPH_OPEN, kernel)
    elif opt == 'c':
        mask = cv.morphologyEx(src, cv.MORPH_CLOSE, kernel)
    if not hide:
        cv.imshow(title, mask)
    if write: cv.imwrite(f'Processing/Post/Processing/Test/Documentation/6. Morfologia/contour_{title}.png', mask)
    return mask

small_kernel = np.ones((2,2),np.uint8)
smedium_kernel = np.ones((3,3),np.uint8)
medium_kernel = np.ones((5,5),np.uint8)

gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray1_blur = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
hand_mask1 = threshold(i, 105, gray1, f'mascara_original', write=True)
hand_mask1 = morph('c', hand_mask1, img, f'mascara_morf_mano_{i}', smedium_kernel, write=True)
hand_mask2 = threshold(j, 105, gray2, f'mascara_original', write=True)
hand_mask2 = morph('c', hand_mask2, img2, f'mascara_morf_mano_{j}', smedium_kernel)
hand_mask2 = morph('o', hand_mask2, img2, f'mascara_morf_mano_{j}', smedium_kernel, write=True)

# ### Brillo y contraste ###

alpha = 1 # Contrast 1
beta = -100 # Brightness -100
hsvEdited = cv.convertScaleAbs(hsvEdited, alpha=alpha, beta=beta)
hsvEditedBlur = cv.convertScaleAbs(hsvEditedBlur, alpha=alpha, beta=beta)

# ### Canales ###

def show_channels(imgs, titles, j):
    for i in range(len(imgs)):
        # cv.imshow(titles[i], imgs[i])
        pass

def split_channels(img, cvt, titles, i):
    img = cv.cvtColor(img, cvt)
    img = cv.bitwise_and(img, img, mask=hand_mask1)
    x, y, z = cv.split(img)
    show_channels([x,y,z], titles, i)
    return img, x, y, z

channel_names = ['hue_editada', 'saturation_editada', 'value_editada', 'hue_editada_blur', 'saturation_editada_blur', 'value_editada_blur']
hsvEdited, hE, sE, vE = split_channels(hsvEdited, cv.COLOR_BGR2HSV, [channel_names[i] for i in range(0, 3)], i)
hsvEditedBlur, hEB, sEB, vEB = split_channels(hsvEditedBlur, cv.COLOR_BGR2HSV, [channel_names[i] for i in range(3, 6)], i)
channel_imgs = [hE, sE, vE, hEB, sEB, vEB]

# ### Histograma ###

def histogram(i, img, channel_name):
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    plt.figure()
    plt.title(f'Histograma {channel_name} imagen {i}')
    plt.xlabel('Intensidad de pixel (0 - 255)')
    plt.ylabel('Número de pixeles')
    plt.grid()
    plt.plot(hist)
    plt.xlim([0,256])

# for j in range(len(channel_imgs)):
#     histogram(i, channel_imgs[j], channel_names[j])

### Contornos ###

lower_hsv_edited = np.array([100, 5, 110], dtype="uint8") # 100, 5, 110
upper_hsv_edited = np.array([150, 180, 255], dtype="uint8") # 150, 180, 255
lower_hsv_edited_blur = np.array([100, 5, 110], dtype="uint8") # 100, 5, 110
upper_hsv_edited_blur = np.array([150, 180, 255], dtype="uint8") # 150, 180, 255

hsv_edited_mask = cv.inRange(hsvEdited, lower_hsv_edited, upper_hsv_edited)
hsv_edited_blur_mask = cv.inRange(hsvEditedBlur, lower_hsv_edited_blur, upper_hsv_edited_blur)

img_blur = createImg(path(i), i)
contour(hsv_edited_mask, img, (255, 255, 0), i, 'HSV_editada', write=True)
contour(hsv_edited_blur_mask, img_blur, (255, 255, 0), i, 'HSV_editada_blur', write=True)

# plt.show()
cv.waitKey()