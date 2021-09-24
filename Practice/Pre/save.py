import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

def video():
    while True:
        _, frame = cap.read()
        cv.imshow('Capture', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return frame

result=cv.imwrite(r'Practice\Pre\1.png', video())
if result==True:
  print('File saved successfully')
else:
  print('error')

cv.destroyAllWindows()
cap.release()