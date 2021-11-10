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

i = input('>> ')

cv.imwrite(r'Processing\Post\Bank\Imgs\\'+f'hand_test_{i}.png', video())

# width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# writer = cv.VideoWriter(r'Practice\Pre\Bank\Vids\\'+f'{i}.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

# while True:
#   ret, frame = cap.read()
#   writer.write(frame)
#   cv.imshow('frame', frame)
#   if cv.waitKey(1) & 0xFF == ord('q'):
#     break

# writer.release()
cap.release()
cv.destroyAllWindows()