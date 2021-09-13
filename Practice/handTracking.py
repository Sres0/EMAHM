import cv2
import mediapipe as mp
import time
import numpy as np

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
                lmList.append([x, y])
                if draw:
                    cv2.circle(img, (x, y), 10, (255, 255, 255), cv2.FILLED)
        return lmList

    def isolate(self, img):
        hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 233], dtype = "uint8")
        upper = np.array([180, 139, 255], dtype = "uint8")
        skinRegionHSV = cv2.inRange(hsvim, lower, upper)
        blurred = cv2.blur(skinRegionHSV, (2,2))
        _,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
        cv2.imshow("thresh", thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = max(contours, key=lambda x: cv2.contourArea(x))
        cv2.drawContours(img, [contours], -1, (255,255,0), 2)
        cv2.imshow("contours", img)

def handTracking():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetection()

    while True:
        _, img = cap.read()
        # img = detector.findHands(img)
        detector.isolate(img)
        # lmList = detector.findPosition(img, draw = False)
        # if len(lmList) != 0:
        #     _width = lmList[17][0] - lmList[5][0] # x positions between little finger and index finger
        #     _len = lmList[0][1] - lmList[12][1] # y positions between wrist and middle finger's tip
        #     print('width: ', _width) 
        #     print('long: ', _len) 
        #     print('estimated rectangular area: ', int(_width * _len))

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 2)
        # cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'): #waitKey(0)
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    handTracking()
    