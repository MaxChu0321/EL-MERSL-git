import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label, perimeter
from .roi import getMask

'''
input:
    image: A 2D gray-scale image.
outputs:
    The value of morphology depending on what function you call.

The functions you can call:
    Area, Perimeter, MajLen_and_MinLen, Eccentricity, Orientation, EquivalentDiameter, Solidity, Extent, 
    Compactness, Rectangularity, Elongation, Roundness, Convexity, ConvexArea, ConvexPerimeter, PerimeterAreaRatio, 
    AspectRatio, MaxRadius, BoundingBoxSize, Ellipticity, DefectsRatio, BendingEnergy, Sphericity, All_Properties
'''

class MorphologyFeaturesExtractor:
    def __init__(self, img:np.array, contour:np.array) -> None:
        self.update(img, contour)

    def update(self, img:np.array, contour:np.array) -> None:
        self._label_img = label( getMask(contour, img.shape).astype("bool") )
        self._stats = regionprops(self.label_img)[0]
        self._contour = np.squeeze(contour)

    @property
    def label_img(self): return self._label_img
    @property
    def stats(self): return self._stats
    @property
    def contour(self): return self._contour   

    def getFeatures(self):
        # Pre-Calculation
        eps = np.spacing(1)
        height = self.label_img.shape[0]
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = self.stats.bbox
        bbox_w = abs(bbox_x2-bbox_x1)
        bbox_h = abs(bbox_y2-bbox_y1)

        area = cv2.contourArea(self.contour)  # 1. Area, 根據 Green 公式計算的，這會考慮到輪廓的形狀和結構 (stas.area 是統計像素)
        imagePerimeter = self.stats.perimeter  # 2. Perimeter
        majLen = self.stats.major_axis_length  # 3. MajorAxis
        minLen = self.stats.minor_axis_length  # 4. MinorAxis
        eccentricity = self.stats.eccentricity  # 5. Eccentricity
        orientation = self.stats.orientation  # 6. Orientation
        equivalentDiameter = self.stats.equivalent_diameter  # 7. EquivDiameter
        solidity = self.stats.solidity  # 8. Solidity
        extent = self.stats.extent  # 9. Extent

        compactness = (imagePerimeter ** 2) / area  # 10. Compactness = perimeter*perimeter / area;
        rectangularity = area / (majLen * minLen)  # 11. Rectangularity(BoundingBox) = area / (major axis * minor axis);
        elongation = 1 - (minLen / majLen)  # 12. Elongation = 1- (minor axis / major axis);
        roundness = 4 * np.pi * area / (imagePerimeter ** 2)  # 13. Roundness
        convexArea = self.stats.convex_area  # 14. ConvexArea
        convexPerimeter = perimeter(self.stats.image_convex)  # 15. Convex_perimeter
        convexity = convexPerimeter / imagePerimeter  # 16. Convexity = Convex_perimeter / perimeter;
        perimeterAreaRatio = area / imagePerimeter  # 17. AP_ratio
        aspectRatio = bbox_h / bbox_w  # 18. Aspect_ratio(AR)

        centers = self.stats.centroid
        radii = np.sqrt( (self.contour[:,0]-centers[1])**2 + (self.contour[:,1]-centers[0])**2 )
        maxRadius = np.amax(radii)
        minRadius = np.amin(radii)
        meanRadii = np.mean(radii)  # 19. Mean of Radii
        boundingBoxSize = bbox_w * bbox_h  # 20. BoundingBoxSize
        ellipticity = (np.pi * majLen ** 2) / (4 * area)  # 21. Ellipticity
        circle_area = np.pi*maxRadius**2  # Circumscribed circle
        defectsRatio = (circle_area-area) / circle_area  # 22. DefectsRatio

        # 把 y 座標轉換到數學坐標系
        self.contour[:, 1] = height - self.contour[:, 1]
        # 計算 x 和 y 的一階和二階有限差分
        dx = np.roll(self.contour[:, 0], -1) - np.roll(self.contour[:, 0], 1)
        dy = np.roll(self.contour[:, 1], -1) - np.roll(self.contour[:, 1], 1)
        dxx = np.roll(self.contour[:, 0], -1) - 2*self.contour[:, 0] + np.roll(self.contour[:, 0], 1)
        dyy = np.roll(self.contour[:, 1], -1) - 2*self.contour[:, 1] + np.roll(self.contour[:, 1], 1)
        # 計算曲率
        curvature = np.abs(dxx * dy - dx * dyy) / (dx * dx + dy * dy + eps)**1.5 # 避免除0
        bendingEnergy = np.sum(curvature**2)  # 23. Bending Energy

        sphericity = minRadius / maxRadius  # 24. Sphericity

        return [area, imagePerimeter, majLen, minLen, eccentricity, orientation, equivalentDiameter, solidity, 
                extent, compactness, rectangularity, elongation, roundness, convexity, convexArea, 
                convexPerimeter, perimeterAreaRatio, aspectRatio, meanRadii, boundingBoxSize, ellipticity, 
                defectsRatio, bendingEnergy, sphericity]
    

    def all(self, feature_prefix=''):
        features =['area', 'perimeter', 'majLen', 'minLen', 'eccentricity', 'orientation', 'equivalentDiameter', 'solidity', 
                    'extent', 'compactness', 'rectangularity', 'elongation', 'roundness', 'convexity', 'convexArea', 
                    'convexPerimeter', 'perimeterAreaRatio', 'aspectRatio', 'meanRadii', 'boundingBoxSize', 'ellipticity', 
                    'defectsRatio', 'bendingEnergy', 'sphericity']
        if feature_prefix != '': 
            for idx, f in enumerate(features): features[idx] = f"{feature_prefix}_{f}"

        feature_values = self.getFeatures()

        return pd.DataFrame(
            [feature_values],
            columns=features
        )
