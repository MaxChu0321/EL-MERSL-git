#
import numpy as np
import pandas as pd
import math
import cv2
import pywt

from .roi import cropInterRect



class WaveletFeaturesExtractor:
    def __init__(self, img:np.array, contour:np.array) -> None:
        self._wt_families = ['db', 'coif', 'sym', 'dmey', 'bior', 'rbio']
        self._wt_levels = 3

        self.update(img, contour)


    def update(self, img:np.array, contour:np.array) -> None:
        self._img = img
        self._contour = contour

        # Cropp image
        self._cropped = cropInterRect(self.img, self.contour)


    @property
    def img(self): return self._img
    @property
    def contour(self): return self._contour
    @property
    def cropped(self): return self._cropped
    
    @property
    def wt_families(self): return self._wt_families

    @property
    def wt_levels(self): return self._wt_levels
    
    @property
    def wt_list(self): 
        wt_list = list()
        for family in self.wt_families:
            wt_list+=[ wt_name for wt_name in pywt.wavelist(family)  ]
        
        return wt_list



    def all(self, feature_prefix=''):
        features = list()
        for wt_name in self.wt_list:
            if feature_prefix != '': wt_name = f"{feature_prefix}_{wt_name}"
            for i in range(self.wt_levels):
                features.append(f"{wt_name}_h{i}")
                features.append(f"{wt_name}_v{i}")
                features.append(f"{wt_name}_d{i}")

        # wavelet transform
        vals = list()
        for wt_name in self.wt_list:
            coeffs = pywt.wavedec2(self.img, wavelet=wt_name, level=self.wt_levels)
            for i in range(self.wt_levels):
                coef_i = coeffs[self.wt_levels-i]
                vals.append(np.log( np.sum(coef_i[0]**2)+np.spacing(1) )) # horizontal
                vals.append(np.log( np.sum(coef_i[1]**2)+np.spacing(1) )) # vertical
                vals.append(np.log( np.sum(coef_i[2]**2)+np.spacing(1) )) # diagonal

        return pd.DataFrame(
            [vals],
            columns=features
        )
