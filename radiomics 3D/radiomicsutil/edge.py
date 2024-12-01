#
import numpy as np
import pandas as pd
import cv2

from .roi import getEdgeMasks
from .grayLevel import getStd, getMean



class EdgeFeaturesExtractor:
    def __init__(self, img:np.array, contour:np.array) -> None:
        self.update(img, contour)


    def update(self, img:np.array, contour:np.array) -> None:
        self._img = img
        self._contour = contour

        ctr_region, in_mask, out_mask = getEdgeMasks(contour, img.shape)
        self._ctr_region = ctr_region
        self._in_mask = in_mask
        self._out_mask = out_mask

    
    @property
    def img(self): return self._img
    @property
    def contour(self): return self._contour
    @property
    def ctr_region(self): return self._ctr_region
    @property
    def in_mask(self): return self._in_mask
    @property
    def out_mask(self): return self._out_mask

    def all(self, feature_prefix=''):
        features = [
            'in_std', 'out_std',
            'in_mean', 'out_mean',
            'std_diff', 'mean_diff',
        ]
        if feature_prefix != '': 
            for idx, f in enumerate(features): features[idx] = f"{feature_prefix}_{f}"

        in_std = getStd(self.img, self.in_mask)
        out_std = getStd(self.img, self.out_mask)
        in_mean = getMean(self.img, self.in_mask)
        out_mean = getMean(self.img, self.out_mask)
        std_diff = abs( in_std-out_std )
        mean_diff = abs( in_mean-out_mean )

        return pd.DataFrame(
            [[in_std, out_std, in_mean, out_mean, std_diff, mean_diff]],
            columns=features
        )


