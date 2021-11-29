import cv2 as cv

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280/3)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720/3)

def video():
    while True:
        _, frame = cap.read()
        cv.line(frame, (0, int(cap.get(4)*4/5)), (cap.get(3), int(cap.get(4)*4/5)), (0, 0, 255), 2) # Línea de muñeca 1/5 de altura img
        cv.imshow('Capture', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return frame

i = input('>> ')

cv.imwrite(r'Processing\Bank\Imgs\\'+f'hand_test_{i}.png', video())

cap.release()
cv.destroyAllWindows()