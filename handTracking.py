import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processedHands = hands.process(rgb)
    # print(processedHands.multi_hand_landmarks)
    if processedHands.multi_hand_landmarks:
        for landmarks in processedHands.multi_hand_landmarks:
            for id, lm in enumerate(landmarks.landmark):
                # print(id, lm)
                h, w, c = img.shape #height, width, channels
                cx, cy = int(lm.x*w), int(lm.y*h) #position for pixel values
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 255, 255), 2, cv2.FILLED)
            mpDraw.draw_landmarks(img, landmarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    # cv2.imshow('RGB', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'): #waitKey(0)
        break

cv2.destroyAllWindows()
cap.release()