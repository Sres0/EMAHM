import cv2 as cv

h = int(720/2)
w = int(1280/2)

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv.CAP_PROP_FRAME_WIDTH, w)

def video(h,w):
    while True:
        _, frame = cap.read()
        cv.line(frame, (0, int(h*4/5)), (w, int(h*4/5)), (0, 0, 255), 2) # Línea de muñeca 1/5 de altura img
        cv.imshow('Capture', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            w = int(1280/3)
            h = int(720/3)
            frame = cv.resize(frame, (w,h))
            return frame

id = input('Identificación: ')

for i in range(1,5):
    cv.imwrite(r'EMAHM\Bank\\'+f'{id}_{i}.png', video(h,w))
    print(f'Toma {i} guardada.')

cap.release()
cv.destroyAllWindows()