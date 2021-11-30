import cv2 as cv

h = int(720/2)
w = int(1280/2)

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
# cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
def show_settings():
    print('*'*5)
    print('brillo: '+str(cap.get(10))) # brillo 8
    print('contraste: '+str(cap.get(11))) # contraste 30
    print('saturación: '+str(cap.get(12))) # sat 50
    print('ganancia: '+str(cap.get(14))) # ganancia 0
    print('exposición: '+str(cap.get(15))) # exposición -5
# cap.set(10, 0) # brillo
# cap.set(11, 20) # contraste
# cap.set(12, 50) # sat
# cap.set(14, 0) # ganancia
# cap.set(15, -4) # exposición
show_settings()

def video(h,w):
    while True:
        _, frame = cap.read()
        cv.line(frame, (0, int(h*4/5)), (w, int(h*4/5)), (0, 0, 255), 2) # Línea de muñeca 1/5 de altura img
        cv.imshow('Capture', frame)
        if cv.waitKey(1) & 0xFF == ord('a'): show_settings()
        if cv.waitKey(1) & 0xFF == ord('q'):
            w = int(1280/3)
            h = int(720/3)
            frame = cv.resize(frame, (w,h))
            return frame

# i = input('>> ')

# cv.imwrite(r'Processing\Bank\Imgs\\'+f'hand_test_{i}.png', video())

# video(h, w)

cap.release()
cv.destroyAllWindows()