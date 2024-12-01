import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from skimage import measure
import matplotlib.pyplot as plt
from IPython.display import display, Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection

from radiomicsutil.voi import find_contours, getMask, calc_vol, calc_mesh, smooth_mesh, plot_mesh, plot_mesh3d



class ThreedMorphologyFeaturesExtractor:
    def __init__(self, voxel_grid:np.array, unit=1, selected:int=0, verbose=False) -> None:
        if verbose: print("Initiating...")
        self._unit = unit
        if verbose: print("Running Marching Cube...")
        verts, faces, _, _ = measure.marching_cubes(voxel_grid, level=0.5)
        self._labeled_mask, self._labels_count = measure.label(voxel_grid, connectivity=2, return_num=True)

        if verbose: print("Calculating Voxels...")
        self._voi_voxels = np.asarray([ np.sum(self.labeled_mask==label) for label in self.labels])

        if verbose: print("Calculating Triangle Mesh...")
        if selected == 0:
            if verbose: print(f"---Label:{self.biggest_voi_label}")
            self._mesh = calc_mesh(self.labeled_mask==self.biggest_voi_label)
        else:
            if verbose: print(f"---Label:{selected}")
            self._mesh = calc_mesh(self.labeled_mask==selected)

        if self.mesh != None:
            if verbose: print("Smoothing Triangle Mesh...")
            self._smoothed_mesh = smooth_mesh(self.mesh)

        if verbose: print("Done.")

    @property
    def unit(self):
        return self._unit
    
    @property
    def labeled_mask(self):
        return self._labeled_mask
    
    @property
    def labels(self):
        return np.arange(1,self._labels_count+1, dtype='int')
    
    @property
    def voi_voxels(self):
        return self._voi_voxels
    
    @property
    def biggest_voi_label(self):
        return np.argmax(self.voi_voxels)+1
    
    @property
    def mesh(self):
        return self._mesh
    
    @property
    def smoothed_mesh(self):
        return self._smoothed_mesh
    
    @staticmethod
    def plot(mesh, color="red", alpha=0.8):
        plot_mesh(mesh, color=color, alpha=alpha)

    @staticmethod
    def quaternion(R:np.array):
        w = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)

        return w, x, y ,z


    def getFeatures(self):
        counts = self._labels_count
        abb = self.mesh.get_axis_aligned_bounding_box()
        convex_hull,_ = self.mesh.compute_convex_hull()

        # Volume
        try:
            volume = self.mesh.get_volume()
        except RuntimeError as e:
            print("Use voxel to estimate volume...")
            volume = np.sum(self.voi_voxels[self.biggest_voi_label-1])
        volume = volume*(self.unit**3)

        # Surface Area
        surface_area = self.smoothed_mesh.get_surface_area()
        surface_area = surface_area*(self.unit**2)

        # Equivalent Diameter
        equivalent_diameter = (6*volume/np.pi)**(1/3)

        # Bounding Box Volume
        abb_volume = abb.volume()*(self.unit**3)
        
        # Extent
        extent = volume / abb_volume

        # Principle Axis Length
        ## 移至座標系中心
        ## 計算協方差矩陣
        ## 計算特徵值和特徵向量
        ## 主軸長度就是特徵值的平方根
        ## 透過 pca 計算即可
        verts = np.asarray(self.mesh.vertices)
        pca = PCA(n_components=3)
        pca.fit(verts)
        principal_axis_length_1, principal_axis_length_2, principal_axis_length_3 = np.sqrt(pca.explained_variance_)*self.unit

        # Orientation
        rotation = pca.components_
        r = R.from_matrix(rotation)
        quaternion_x, quaternion_y ,quaternion_z, quaternion_w = r.as_quat()
        euler_roll, euler_pitch, euler_yaw = r.as_euler("xyz")
        
        # Eccentricity
        ## 取 principal_length1-3平面上的 eccentricity, 一定比 1-2 平面大
        ## eccentricity = np.sqrt((principal_axis_length_1/2)**2-(principal_axis_length_3/2)**2)/principal_axis_length_1 # from yijen, 分母應為 principal_axis_length_1/2
        ## 分子分母上下抵銷，所以直徑(主軸長)半徑不影響
        eccentricity = np.sqrt( 1-(principal_axis_length_3**2/principal_axis_length_1**2) )

        # Convex Hull Volume
        convex_volume = convex_hull.get_volume()*(self.unit**3)

        # Solidity
        solidity = volume / convex_volume

        # Convex Hull Surface Area
        convex_area = convex_hull.get_surface_area()*(self.unit**2)

        # Convexity
        convexity = convex_area / surface_area

        # Compactness
        compactness = surface_area**3/volume**2

        # Rectangularity ## need to discuss
        rectangularity = surface_area/((principal_axis_length_1*principal_axis_length_2+principal_axis_length_1*principal_axis_length_3+principal_axis_length_2*principal_axis_length_3)*2) # from yijen
        # rectangularity = volume / (principal_axis_length_1*principal_axis_length_2*principal_axis_length_3)

        # Elongation
        elongation = 1-(principal_axis_length_3/principal_axis_length_1)

        # Sphericity Roundness # same?
        sphericity_roundness = (4*np.pi*(equivalent_diameter/2)**2)/surface_area

        # Sphericity Ellipticity # same?
        ## sphericity_ellipticity = (np.pi**(1/3)*(6*volume)**(2/3))/surface_area # from yijen
        a = principal_axis_length_1/2
        c = principal_axis_length_3/2
        oblate_spheroid_surface_area = (2*np.pi*a**2) + np.pi * (c**2/eccentricity) * np.log( (1+eccentricity/1-eccentricity) )
        sphericity_ellipticity = oblate_spheroid_surface_area/surface_area

        # Area Volume Ratio
        area_volume_ratio = surface_area/volume

        # Aspect Ratio 1 2 3
        aspect_ratio_1 = principal_axis_length_2 / principal_axis_length_1
        aspect_ratio_2 = principal_axis_length_3 / principal_axis_length_1
        aspect_ratio_3 = principal_axis_length_3 / principal_axis_length_2

        # Max Radius
        centroid = np.mean(verts, axis=0)
        distances = np.linalg.norm(verts-centroid, axis=1)
        max_radius = np.max(distances)*self.unit

        # Defect Ratio
        defect_ratio = ((4/3)*np.pi*(max_radius)**3-volume)/((4/3)*np.pi*(max_radius)**3)

        # Gaussain Curvature Mean/Mean Curvature Mean: 
        ## 當你在 mesh 上的某個頂點使用一個半徑為 r 的球來計算曲率時，這個球可能會與 mesh 有更大或更小的交集面積。
        ## 這個交集面積會隨著 r 的變化而變化。為了得到一個獨立於 r 的曲率估計，你可以將計算出的曲率除以這個交集面積。
        sb_radius=self.unit
        tri_mesh = trimesh.Trimesh(vertices=np.asarray(self.smoothed_mesh.vertices), faces=np.asarray(self.smoothed_mesh.triangles))
        gauss_curv_mean = np.mean( discrete_gaussian_curvature_measure(tri_mesh, tri_mesh.vertices, sb_radius)/sphere_ball_intersection(sb_radius, sb_radius) )
        mean_curv_mean = np.mean( discrete_mean_curvature_measure(tri_mesh, tri_mesh.vertices, sb_radius)/sphere_ball_intersection(sb_radius, sb_radius) )

        return [
            counts, volume, surface_area, equivalent_diameter, extent,
            principal_axis_length_1, principal_axis_length_2, principal_axis_length_3,
            quaternion_w, quaternion_x, quaternion_y, quaternion_z,
            euler_roll, euler_pitch, euler_yaw,
            eccentricity, solidity, convex_volume, convex_area, convexity,
            compactness, rectangularity, elongation, sphericity_roundness,
            area_volume_ratio, aspect_ratio_1, aspect_ratio_2, aspect_ratio_3,
            max_radius, abb_volume, sphericity_ellipticity, defect_ratio,
            gauss_curv_mean, mean_curv_mean
        ]
    

    def all(self, feature_prefix=''):
        features = [
            'counts', 'volume', 'surface_area', 'equivalent_diameter', 'extent',
            'principal_axis_length_1', 'principal_axis_length_2', 'principal_axis_length_3',
            'quaternion_w', 'quaternion_x', 'quaternion_y', 'quaternion_z',
            'euler_roll', 'euler_pitch', 'euler_yaw',
            'eccentricity', 'solidity', 'convex_volume', 'convex_area', 'convexity',
            'compactness', 'rectangularity', 'elongation', 'sphericity_roundness',
            'area_volume_ratio', 'aspect_ratio_1', 'aspect_ratio_2', 'aspect_ratio_3',
            'max_radius', 'abb_volume', 'sphericity_ellipticity', 'defect_ratio',
            'gauss_curv_mean', 'mean_curv_mean'
        ]
        if feature_prefix != '': 
            for idx, f in enumerate(features): features[idx] = f"{feature_prefix}_{f}"

        feature_values = self.getFeatures()

        return pd.DataFrame(
            [feature_values],
            columns=features
        )