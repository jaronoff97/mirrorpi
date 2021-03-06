import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.rectangle(img, (300, 300), (0, 0), (0, 255, 0), 0)
    roi = img[0:300, 0:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    _, edges = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(
        edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    y = len(contours)
    area = np.zeros(y)
    for i in range(0, y):
        area[i] = cv2.contourArea(contours[i])

    index = area.argmax()
    hand = contours[index]
    x, y, w, h = cv2.boundingRect(hand)
    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 0)
    temp = np.zeros(roi.shape, np.uint8)

    hull = cv2.convexHull(hand)
    cv2.drawContours(temp, [hand], -1, (0, 255, 0), 0)
    cv2.drawContours(temp, [hull], -1, (0, 0, 255), 0)
    hull = cv2.convexHull(hand, returnPoints=False)
    defects = cv2.convexityDefects(hand, hull)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        # start, end, farthest point, approx distance to farthest point of
        # defect
        start = tuple(hand[s][0])
        end = tuple(hand[e][0])
        far = tuple(hand[f][0])
        cv2.line(temp, far, start, [0, 255, 0], 2)
        cv2.circle(temp, far, 3, [0, 0, 255], -1)

    print(range(defects.shape[0]))

    cv2.drawContours(edges, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Place your hand in the rectangle', img)
    cv2.imshow('B', np.hstack((temp, roi)))
    cv2.moveWindow('B', 500, 300)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
