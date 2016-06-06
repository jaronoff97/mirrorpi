import numpy as np
import cv2


def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('Image')
cv2.createTrackbar('LOW', 'Image', 0, 255, nothing)
cv2.createTrackbar('HIGH', 'Image', 0, 255, nothing)


while(cap.isOpened()):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    low = cv2.getTrackbarPos('LOW', 'Image')
    high = cv2.getTrackbarPos('HIGH', 'Image')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, low, high, 1)

    contours, h = cv2.findContours(thresh, 1, 2)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        # print(len(approx))
        if len(approx) == 5:
            # print("pentagon")
            cv2.drawContours(img, [cnt], 0, 255, -1)
        elif len(approx) == 3:
            # print("triangle")
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), -1)
        elif len(approx) == 4:
            # print("square")
            cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)
        elif len(approx) == 9:
            # print("half-circle")
            cv2.drawContours(img, [cnt], 0, (255, 255, 0), -1)
        elif len(approx) > 15:
            # print("circle")
            cv2.drawContours(img, [cnt], 0, (0, 255, 255), -1)

    cv2.imshow('Image', img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
