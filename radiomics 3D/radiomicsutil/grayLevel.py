import pandas as pd
import numpy as np
from .radiomics import *
from scipy.stats import skew, kurtosis
from math import sqrt

from .roi import getMask
'''
input:
    image: A 2D gray-scale image.
outputs:
    The value of gray-scale color feature depending on what function you call.

The functions you can call:
    Mean, Median, Variance_and_StdDev, MaxPixel_and_MinPixel, Skewness, Kurtosis, Energy, Entropy, All_Properties
'''

'''
The following functions calculate the gray-scale color features of input image respectively.
You can call the function "All_Properties" to get all the features of input image at one time.
The output of the function "All_Properties" is in a list format.
'''

def getMax(img: np.array, mask: np.array) -> float:
    '''計算 roi 的最大像素值
    Args:
        img: original grayscale image
        mask: mask of roi (1->True, 0->False)
    Returns:
        max: the maximum value of pixles in roi
    '''
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")
    
    masked_img = np.where(mask == 1, img, np.nan)
    return np.nanmax(masked_img)


def getMin(img: np.array, mask: np.array) -> float:
    '''計算 roi 的最小像素值
    Args:
        img: original grayscale image
        mask: mask of roi (1->True, 0->False)
    Returns:
        min: the minimum value of pixles in roi
    '''
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")
    
    masked_img = np.where(mask == 1, img, np.nan)
    return np.nanmin(masked_img)


def getMedian(img: np.array, mask: np.array) -> float:
    '''計算 roi 的像素中的灰階中位數
    Args:
        img: original grayscale image
        mask: mask of roi (1->True, 0->False)
    Returns:
        med: the median value of pixles in roi
    '''
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")
    
    masked_img = np.where(mask == 1, img, np.nan)
    return np.nanmedian(masked_img)


def getVar(img: np.array, mask: np.array) -> float:
    '''計算 roi 的變異數
    Args:
        img: original grayscale image
        mask: mask of roi (1->True, 0->False)
    Returns:
        var: the variance value of pixles in roi
    '''
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")
    
    masked_img = np.where(mask == 1, img, np.nan)
    return np.nanvar(masked_img)


def getStd(img: np.array, mask: np.array) -> float:
    '''計算 roi 的標準差
    Args:
        img: original grayscale image
        mask: mask of roi (1->True, 0->False)
    Returns:
        std: standard deviation
    '''
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")
    
    masked_img = np.where(mask == 1, img, np.nan)
    return np.nanstd(masked_img)


def getMean(img: np.array, mask: np.array) -> float:
    '''計算 roi 的平均
    Args:
        img: original grayscale image
        mask: mask of roi (1->True, 0->False)
    Returns:
        mean: mean
    '''
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")
    
    masked_img = np.where(mask == 1, img, np.nan)
    return np.nanmean(masked_img)


def getSkewness(img: np.array, mask: np.array) -> float:
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")

    masked_img = np.where(mask == 1, img, np.nan)
    skewness = skew(masked_img.flatten(), nan_policy="omit")
    return skewness if not np.isnan(skewness) else 0


def getKurtosis(img: np.array, mask: np.array) -> float:
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")

    masked_img = np.where(mask == 1, img, np.nan)
    kurtosis_val = kurtosis(masked_img.flatten(), nan_policy="omit")
    return kurtosis_val if not np.isnan(kurtosis_val) else 0


def getEnergy(img: np.array, mask: np.array) -> float:
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")

    masked_img = np.where(mask == 1, img, np.nan)
    return sqrt(np.nanmean(masked_img**2))

def getEntropy(img: np.array, mask: np.array) -> float:
    if img.shape != mask.shape: raise ValueError("sizes of image and mask are not match")

    # Extract ROI
    masked_img = np.where(mask == 1, img, np.nan)
    # Flatten pixels
    filtered_arr = masked_img[~np.isnan(masked_img)].flatten()
    # Calculate grayscale histogram
    hist, _ = np.histogram(filtered_arr, bins=256, range=[0, 256])
    # Calculate amount of pixels
    N = len(filtered_arr)
    # Calculate probabilities of every grayscale
    probabilities = hist / N
    # Calculate the entropy
    nonzero_probabilities = probabilities[probabilities != 0]
    entropy = -np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))
    return entropy


class GrayLevelFeaturesExtractor:
    def __init__(self, img:np.array, contour:np.array) -> None:
        self.update(img, contour)


    def update(self, img:np.array, contour:np.array) -> None:
        self._img = img
        self._contour = contour
        self._mask = getMask(self.contour, self.img.shape)

        # ctr_region, in_mask, out_mask = getEdgeMasks(contour, img.shape)
        # self._ctr_region = ctr_region
        # self._in_mask = in_mask
        # self._out_mask = out_mask

    
    @property
    def img(self): return self._img
    @property
    def contour(self): return self._contour
    @property
    def mask(self): return self._mask

    def all(self, feature_prefix=''):
        features = [
            'mean', 'median',
            'variance', 'stdDev',
            'maxPixel', 'minPixel',
            'skewness', 'kurtosis',
            'energy', 'entropy',
        ]
        if feature_prefix != '': 
            for idx, f in enumerate(features): features[idx] = f"{feature_prefix}_{f}"
        mean = getMean(self.img, self.mask)
        median = getMedian(self.img, self.mask)
        variance = getVar(self.img, self.mask)
        stdDev = getStd(self.img, self.mask)
        maxPixel = getMax(self.img, self.mask)
        minPixel = getMin(self.img, self.mask)
        skewness = getSkewness(self.img, self.mask)
        kurtosis = getKurtosis(self.img, self.mask)
        energy = getEnergy(self.img, self.mask)
        entropy = getEntropy(self.img, self.mask)

        return pd.DataFrame(
            [[mean, median, variance, stdDev, maxPixel, minPixel, skewness, kurtosis, energy, entropy]],
            columns=features
        )