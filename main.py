import cv2
import mediapipe as mp  # type: ignore
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand_LMS in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_LMS.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, hand_LMS, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
