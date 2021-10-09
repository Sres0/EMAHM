import cv2 as cv

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

def video():
    while True:
        _, frame = cap.read()
        cv.imshow('Capture', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return frame

i = 25.3

result=cv.imwrite(r'Practice\Pre\\'+f'{i}.png', video())
if result==True:
  print('File saved successfully')
else:
  print('error')

cv.destroyAllWindows()
cap.release()