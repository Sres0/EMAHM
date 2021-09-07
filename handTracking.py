import cv2
import mediapipe as mp
import time

class HandDetection():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.processedHands = self.hands.process(rgb)
        if self.processedHands.multi_hand_landmarks:
            if draw:
                for landmarks in self.processedHands.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum = 0, draw = True): #only one hand
        lmList = []
        if self.processedHands.multi_hand_landmarks:
            hand = self.processedHands.multi_hand_landmarks[handNum]
            for id, lm in enumerate(hand.landmark):
                h, w, _ = img.shape
                x, y = int(lm.x*w), int(lm.y*h)
                lmList.append([id, x, y])
                if draw:
                    cv2.circle(img, (x, y), 10, (255, 255, 255), cv2.FILLED)
        return lmList

def handTracking():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetection()

    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw = False)
        if len(lmList) != 0:
            print(lmList[12]) # index tip positions

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 2)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'): #waitKey(0)
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    handTracking()