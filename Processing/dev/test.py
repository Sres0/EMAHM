import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

i = 100

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

img = createImg(i, hide=True)
img2 = img.reshape((-1,3))

from sklearn.mixture import GaussianMixture as GMM
gmm_model = GMM(3, covariance_type='tied').fit(img2)
gmm_labels = gmm_model.predict(img2)
original_shape = img.shape
segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
cv.imshow('image', img)
plt.figure()
plt.imshow(segmented)
plt.show()
# segmented = cv.integral(np.uint8(segmented))
# segmented = cv.normalize(segmented, None, 255,0, cv.NORM_MINMAX, cv.CV_8UC1)
# cv.imshow('segmented', segmented)

cv.waitKey(0) & 0xFF=='q'
cv.destroyAllWindows()