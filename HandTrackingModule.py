# This script enables the webcam
# This module gets the position of the hand landmarks for the user easily
import cv2
import mediapipe as mp
import time
import math


class HandDetector():
    def __init__(self, mode=False,  maxHands=2, complexity=1, detectionConf=0.5, trackConf=0.5):
        # self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        # Create an object called hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionConf, self.trackConf)
        # for connecting all the hand points detected
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            # extract the information for each hand that is detected
            for handLms in self.results.multi_hand_landmarks:
                # for a single hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        # List containing all values of X
        xList = []
        # List containing all values of Y
        yList = []
        boundingbox = []
        # list with all the landmarks positions
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                height, width, c = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    # if id == 4:  # for choosing which landmark gets highlighted
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin,xmax = min(xList),max(xList)
            ymin,ymax = min(yList), max(yList)
            # boundingbox contains the values of interest
            boundingbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (boundingbox[0]-20, boundingbox[1]-20),
                (boundingbox[2]+20, boundingbox[3]+20), (0, 255, 0), 2)

        return self.lmList, boundingbox

    def fingersUp(self):
        # list of five fingers 1 or 0 value
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                # return 5 values representing whether the finger is up or not
        return fingers

    def findDistance(self,p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        # add the argument draw=False in the two functions below in order to disable drawing
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            # gets the landmark no specified and prints it if the list has something
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
