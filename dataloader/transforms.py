
import numpy as np
import random
import torch
import copy

from PIL import Image


class LoadImage():
    def __init__(self, target1="left_image_path", target2="right_image_path", root_path=''):
        self.target1 = target1
        self.target2 = target2
        self.root_path = root_path
        """
        Load, organize channels, and standardize intensity of images.
        """

    def __call__(self, data):
        # Read image
        img = np.array(Image.open(self.root_path + data[self.target1]), dtype=float)
        if np.max(img) > 1:
            img /= 255

        # channel first
        if len(img.shape) > 2:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, 0)

        data[self.target1.replace("_path", "")] = img

        # Read image
        img = np.array(Image.open(self.root_path + data[self.target2]), dtype=float)
        if np.max(img) > 1:
            img /= 255

        # channel first
        if len(img.shape) > 2:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, 0)

        data[self.target2.replace("_path", "")] = img
        return data


class CopyDict():
    def __call__(self, data):
        d = copy.deepcopy(data)
        return d
