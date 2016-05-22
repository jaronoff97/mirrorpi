import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg2 = cv2.bgsegm.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg2.apply(frame)

    cv2.imshow('frame', fgmask)
    cv2.imshow('frame2', fgmask2)
    k = cv2.waitKey(10) & 0xff
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
