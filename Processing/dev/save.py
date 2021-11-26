import cv2 as cv

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

def video():
    while True:
        _, frame = cap.read()
        cv.line(frame, (0, int(720*4/5)), (1280, int(720*4/5)), (0, 0, 255), 2) # Línea de muñeca 1/5 de altura img
        cv.imshow('Capture', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            w = int(1280/3)
            h = int(720/3)
            frame = cv.resize(frame, (w,h))
            return frame

i = input('>> ')

cv.imwrite(r'Processing\Bank\Imgs\\'+f'hand_test_{i}.png', video())

cap.release()
cv.destroyAllWindows()