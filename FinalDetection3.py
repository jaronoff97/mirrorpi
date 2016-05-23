import cv2
import numpy as np
import math

cascPath = '/Users/jea/Documents/Code/python/opencv/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


def getFaces(grey):
    return faceCascade.detectMultiScale(
        grey,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


def get_contour(contours):
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    return cnt


def draw_drawing(cnt, hull, img):
    drawing = np.zeros(img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
    cv2.imshow('drawing', drawing)


def draw_contours(drawing, img):
    all_img = np.hstack((drawing, img))
    cv2.imshow('Contours', all_img)


def draw_thresh(thresh, contours):
    cv2.drawContours(thresh, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Thresholded', thresh)


def count_fingers(cnt, img):
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
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
            cv2.circle(img, far, 1, [0, 0, 255], -1)
        # dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(img, start, end, [0, 255, 0], 2)
        # cv2.circle(crop_img,far,5,[0,0,255],-1)
    # draw_drawing(cnt, hull, img)
    return count_defects


def apply_filter(img, apply_gauss=False):
    def gaussian(img):
        value = (35, 35)
        return cv2.GaussianBlur(img, value, 0)
    final_img = (img if apply_gauss is True else gaussian(img))
    return cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)


def brighter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert it to hsv
    h, s, v = cv2.split(hsv)
    v += 255
    final_hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def removeBG(img):
    fgmask = fgbg.apply(brighter(img))
    res = cv2.bitwise_and(img, img, mask=fgmask)
    cv2.imshow('fg', res)


def main():
    while(cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        removeBG(img)

        cv2.rectangle(img, (300, 300), (50, 50), (0, 255, 0), 0)
        crop_img = img[50:300, 50:300]
        cropped_grey = apply_filter(crop_img, True)
        uncropped_grey = apply_filter(img)
        # Draw a rectangle around the faces
        for (x, y, w, h) in getFaces(uncropped_grey):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        _, thresh1 = cv2.threshold(cropped_grey, 127, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(thresh1.copy(),
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_NONE)
        cnt = get_contour(contours)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        fingers = count_fingers(cnt, crop_img)
        cv2.putText(img, "{0} Fingers".format(fingers), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))

        # cv2.imshow('end', crop_img)
        cv2.imshow('Video', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
