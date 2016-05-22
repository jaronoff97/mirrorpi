import cv2
import numpy as np
cam = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while(cam.isOpened):
    f, img = cam.read()
    if f is True:
        # img=cv2.flip(img,1)
        # img=cv2.medianBlur(img,3)
        fgmask = fgbg.apply(img)
        cv2.imshow('track', fgmask)
        cv2.imshow('img', img)
    if(cv2.waitKey(27) != -1):
        cam.release()
        cv2.destroyAllWindows()
        break

