import cv2
import numpy as np
import math


class FaceFinder(object):
    """docstring for FaceFinder"""

    def __init__(self):
        super(FaceFinder, self).__init__()
        self.cascPath = '''/Users/jea/Documents/Code/python/opencv/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml'''
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.hundred_image = cv2.imread(
            "100Dollar.png", cv2.CV_LOAD_IMAGE_UNCHANGED)
        self.hundred_copy = self.hundred_image.copy()

    def getFaces(self, grey):
        return self.faceCascade.detectMultiScale(
            grey,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    def drawImageOnImage(self, toPut, destination, x_offset, y_offset):
        try:
            destination[y_offset:y_offset + toPut.shape[0],
                        x_offset:x_offset + toPut.shape[1]] = toPut
        except Exception, e:
            print(e)
        else:
            print("All good")

    def makeOverlay(self, toPut, destination, x_offset, y_offset):
        top_left = (x_offset, y_offset)
        bot_right = (x_offset + toPut.shape[1], y_offset + toPut.shape[0])
        # print("TOP LEFT: ", top_left)
        # print("BOT RIGHT: ", bot_right)
        # print("Shape: ", toPut.shape)
        try:
            for c in range(0, 3):
                destination[top_left[1]:bot_right[1], top_left[0]:bot_right[0], c] = toPut[:, :, c] * (
                    toPut[:, :, 3] / 255.0) + destination[top_left[1]:bot_right[1], top_left[0]:bot_right[0], c] * (1.0 - toPut[:, :, 3] / 255.0)
        except Exception, e:
            print(e)
        else:
            print("All good")

    def drawFaces(self, img, destination):
        for (x, y, w, h) in self.getFaces(img):
            self.hundred_image = cv2.resize(
                self.hundred_copy, (w * 2, int(h)),
                interpolation=cv2.INTER_NEAREST)
            self.makeOverlay(self.hundred_image,
                             destination, (x - w / 2), (y))
            # cv2.rectangle(destination, (x, y),
            #  (x + w, y + h),
            #  (0, 255, 0), 2)
