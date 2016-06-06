import numpy as np
import cv2
import math


class FingerFinder(object):
    """docstring for FingerFinder"""

    def __init__(self, background_reduction=False):
        super(FingerFinder, self).__init__()
        self.bg_reduction = background_reduction
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    def process(self, frame):
        res = cv2.flip(frame, 1)
        if self.bg_reduction:
            res = self.apply_mask(res)
        fingers1 = self.find_hand(res, 50, 50, 300, 450, 1)
        fingers2 = self.find_hand(res, 50, 590, 900, 450, 2)
        self.draw_bounding(res, 50, 50, 300, 450, fingers1)
        self.draw_bounding(res, 590, 50, 900, 450, fingers2)

    def find_border(self, contours):
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        return cnt

    def apply_mask(self, frame):
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.dilate(fgmask, self.kernel, iterations=10)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        return cv2.bitwise_and(frame, frame, mask=fgmask)

    def make_thresh(self, res):
        grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, thresh1 = cv2.threshold(grey, 127, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(thresh1.copy(),
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_NONE)
        return contours

    def count_fingers(self, cnt, img):
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
        return count_defects

    def find_hand(self, res, xpos, ypos, width, height, handnum):
        crop_img = res[xpos:height, ypos:width]
        cv2.imshow('hand {0}'.format(handnum), crop_img)
        contours = self.make_thresh(crop_img)
        cnt = self.find_border(contours)
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        #
        #
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
        fingers = self.count_fingers(cnt, crop_img)
        return fingers

    def draw_bounding(self, res, xpos, ypos, width, height, fingers):
        cv2.rectangle(res, (width, height), (xpos, ypos), (0, 255, 0), 0)
        cv2.putText(res, "{0} Fingers".format(fingers), (xpos, ypos),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))