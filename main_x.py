import HandTrackingModule as htm
import cv2
import mediapipe as mp 

cap = cv2.VideoCapture(0)
detector = htm.handDetector(max_hands=4)

while True:
    success, img = cap.read()
    img = detector.findHands(img, True)
    lmlist = detector.findPos(img, 0)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


def fingers_up(self):
    fingers = []

    if self.lmlist[self.tip_ids[0]][1] > self.lmlist[self.tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):
        if self.lmlist[self.tip_ids[id]][2] < self.lmlist[self.tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers
