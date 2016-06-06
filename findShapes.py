# import the necessary packages
import numpy as np
import argparse
import cv2


def nothing(x):
    pass
cap = cv2.VideoCapture(0)
cv2.namedWindow('Image')
cv2.createTrackbar('LOW', 'Image', 0, 255, nothing)
cv2.createTrackbar('HIGH', 'Image', 0, 255, nothing)

while(cap.isOpened()):
    # load the image
    ret, image = cap.read()
    image = cv2.flip(image, 1)
    low = cv2.getTrackbarPos('LOW', 'Image')
    high = cv2.getTrackbarPos('HIGH', 'Image')
    # find all the 'black' shapes in the image
    lower = np.array([low, low, low])
    upper = np.array([high, high, high])
    shapeMask = cv2.inRange(image, lower, upper)

    # find the contours in the mask
    (cnts, _) = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    print "I found %d black shapes" % (len(cnts))
    # cv2.imshow("Mask", shapeMask)

    # loop over the contours
    for c in cnts:
        # draw the contour and show it
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
