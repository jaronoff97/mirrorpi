import numpy as np
import cv2
import math
import ImageHelpers
from FingerFinder import FingerFinder
from FaceFinder import FaceFinder
import MazeMaker

def nothing(x):
    pass


def main(camera):
    faceFinder = FaceFinder()
    # fingerFinder = FingerFinder()
    # maze_image = MazeMaker.makeBlankImage()
    # MazeMaker.generate_maze_helper(maze_image)
    while(camera.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        grey = ImageHelpers.apply_filter(img)
        grey = ImageHelpers.brighter(grey, 3.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        faceFinder.drawFaces(grey, img)
        cv2.imshow('Video', img)
        # cv2.imshow('Grey', grey)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    main(cap)
