import cv2
import mediapipe as mp 

class handDetector:
    def __init__(self, mode=False, max_hands=2, detection_conf=0.8, track_conf=0.5):
        self.mode = mode
        self.maxhands = max_hands
        self.detectionConf = detection_conf
        self.trackConf = track_conf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands)
        self.mpDraw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand_LMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, hand_LMS, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPos(self, img, handno=0, draw=False):
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmlist

    def fingers_up(self):
        fingers = []

        if self.tip_ids[0] < len(self.lmlist) and self.tip_ids[0] - 1 < len(
            self.lmlist
        ):
            if self.lmlist[0][1] > self.lmlist[4][1]:
                if (
                    self.lmlist[self.tip_ids[0]][1]
                    < self.lmlist[self.tip_ids[0] - 1][1]
                ):
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if (
                    self.lmlist[self.tip_ids[0]][1]
                    > self.lmlist[self.tip_ids[0] - 1][1]
                ):
                    fingers.append(1)
                else:
                    fingers.append(0)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.tip_ids[id] < len(self.lmlist) and self.tip_ids[id] - 2 < len(
                self.lmlist
            ):
                if (
                    self.lmlist[self.tip_ids[id]][2]
                    < self.lmlist[self.tip_ids[id] - 2][2]
                ):
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                fingers.append(0)

        return fingers


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    lmlist = []
    while True:
        success, img = cap.read()
        img = detector.findHands(img, True)
        lmlist = detector.findPos(img, 0)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
if __name__ == '__main__':
    main()