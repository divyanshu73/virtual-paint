import cv2
import os
import numpy as np
import time
import HandTrackingModule as htm

folder_path = "header_images"
header_img_path = os.listdir(folder_path)
header_images = []
color = (0, 0, 0)
size = 15
brush_thickness = size
eraser_thickness = 80
xp, yp = 0, 0


for img_path in header_img_path:
    image = cv2.imread(f"{folder_path}/{img_path}")
    image_resized = cv2.resize(image, (1280, 125))
    header_images.append(image_resized)

header = header_images[0]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

img_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    landmark_list = detector.findPos(img)

    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        fingers = detector.fingers_up()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("Selection Mode")
            if y1 < 125:
                if 0 < x1 < 150:
                    header = header_images[1]
                    color = (0, 0, 255)
                    size = 10
                elif 180 < x1 < 330:
                    header = header_images[2]
                    color = (255, 0, 0)
                    size = 10
                elif 380 < x1 < 530:
                    header = header_images[3]
                    color = (0, 255, 255)
                    size = 10
                elif 610 < x1 < 760:
                    header = header_images[4]
                    color = (0, 255, 0)
                    size = 10
                elif 830 < x1 < 980:
                    header = header_images[5]
                    color = (147, 20, 255)
                    size = 10
                elif 1040 < x1 < 1280:
                    header = header_images[6]
                    color = (0, 0, 0)
                    size = 20
            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), color, cv2.FILLED)

        else:
            # print("Drawing Mode")
            cv2.circle(img, (x1, y1), size, color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color == (255, 255, 255):
                cv2.line(img, (xp, yp), (x1, y1), color, eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), color, eraser_thickness)

            cv2.line(img, (xp, yp), (x1, y1), color, brush_thickness)
            cv2.line(img_canvas, (xp, yp), (x1, y1), color, brush_thickness)
            xp, yp = x1, y1

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    img[0:125, 0:1280] = header
    # img=cv2.addWeighted(img,0.5,img_canvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", img_canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # cv2.destroyAllWindows()
        break
