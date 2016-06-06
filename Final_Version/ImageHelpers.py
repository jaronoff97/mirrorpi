from __future__ import print_function
import numpy as np
import argparse
import cv2


def apply_filter(img, apply_gauss=False):
    def gaussian(img):
        value = (35, 35)
        return cv2.GaussianBlur(img, value, 0)
    final_img = (img if apply_gauss is True else gaussian(img))
    return cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)


def crop(img, x, y, w, h):
    return img[x:w, y:h]


def brighter(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
