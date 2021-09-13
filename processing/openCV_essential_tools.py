import cv2 as cv

img = cv.imread('images/UV_img_sample.jpg')
cv.imshow('UV', img)

### GRAYSCALE ###
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('B&W', gray)

### BLUR ###
# blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT) #Column must be odd number
# blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT) #Way more blurred
# cv.imshow('Blur', blur)

### EDGE CASCADE ###
# canny = cv.Canny(img, 100, 150)
# canny = cv.Canny(blur, 50, 150) #Passing the blur to reduce the edges
# cv.imshow('Canny', canny)

### Dilate ###
# dilated = cv.dilate(canny, (7,7), iterations=3) # ?
# cv.imshow('Dilated', dilated)

### Erode ###
# eroded = cv.erode(canny, (7,7), iterations=3)
# cv.imshow('Eroded', eroded) #Reversing dilated

### Resize ###
# resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA) #For < dimensions than the original
# resized = cv.resize(img, (950, 950), interpolation=cv.INTER_LINEAR) #For > dimensions than the original. Also cubic but takes more effort
# cv.imshow('Resized', resized)

### CROP ###
crop = img[50:200, 300:400]
cv.imshow('Cropped', crop)

cv.waitKey(0)