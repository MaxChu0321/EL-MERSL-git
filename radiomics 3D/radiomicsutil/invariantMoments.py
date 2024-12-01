import cv2
import numpy as np
import pandas as pd

from .roi import cropInterRect

'''
input:
    image: A 2D gray-scale image.
output:
    The seven values of Hu moment invariants, which is in an array format.

The functions you can call:
    InvariantMoments
'''

def getInvariantMoments(img: np.array) -> np.array:
    moments = cv2.moments(img)
    huMoments = cv2.HuMoments(moments).flatten()
    return huMoments



class InvariantMomentsFeaturesExtractor:
    def __init__(self, img:np.array, contour:np.array) -> None:
        self.update(img, contour)


    def update(self, img:np.array, contour:np.array) -> None:
        self._img = img
        self._contour = contour
        self._cropped = cropInterRect(self.img, self.contour)


    @property
    def img(self): return self._img
    @property
    def contour(self): return self._contour
    @property
    def cropped(self): return self._cropped


    def all(self, feature_prefix=''):
        features = [
            'hu-0', 'hu-1',
            'hu-2', 'hu-3',
            'hu-4', 'hu-5',
            'hu-6'
        ]
        if feature_prefix != '': 
            for idx, f in enumerate(features): features[idx] = f"{feature_prefix}_{f}"
        
        huMoments = getInvariantMoments(self.cropped)

        return pd.DataFrame(
            [huMoments],
            columns=features
        )