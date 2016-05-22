import cv2
import numpy as np
import math

cascPath = '/Users/jea/Documents/Code/python/opencv/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# - - - - - - - - - - - - - - - - - - - - - - - -kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# - - - - - - - - - - - - - - - - - - - - - - - -fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while(cap.isOpened()):
    ret, img = cap.read()
    fgmask = fgbg.apply(img)

    cv2.rectangle(img, (300, 300), (50, 50), (0, 255, 0), 0)
    crop_img = img[50:300, 50:300]
    cropped_grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    uncropped_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        uncropped_grey,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    value = (35, 35)
    blurred = cv2.GaussianBlur(cropped_grey, value, 0)
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_NONE)
    max_area = -1
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if(area > max_area):
            max_area = area
            ci = i
    cnt = contours[ci]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
        # dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(crop_img, start, end, [0, 255, 0], 2)
        # cv2.circle(crop_img,far,5,[0,0,255],-1)
    cv2.putText(img, "{0} Fingers".format(count_defects), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))

    cv2.imshow('fg', fgmask)
    # cv2.imshow('end', crop_img)
    # cv2.imshow('drawing', drawing)
    # cv2.imshow('Thresholded', thresh1)
    cv2.imshow('Video', img)
    all_img = np.hstack((drawing, crop_img))
    # cv2.imshow('Contours', all_img)
    k = cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
