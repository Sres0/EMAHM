import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv.imshow('UV', frame)
    if cv.waitKey(2) & 0xFF == ord('q'): #waitKey(0)
        break