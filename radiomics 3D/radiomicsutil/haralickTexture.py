#
import numpy as np
import pandas as pd
import math
import cv2
from skimage.exposure import rescale_intensity
from skimage.feature import graycomatrix, graycoprops

from .roi import cropInterRect



class HaralickTextureFeaturesExtractor:
    def __init__(self, img:np.array, contour:np.array) -> None:
        self._glcm_settings = {
            'num_levels': 32,
            'distances': {
                '1':1,'2':2,'3':3,'4':4,'5':5
            },
            'angles': {
                '0':0, '45': np.pi/4, '90':np.pi/2, '135':3*np.pi/4
            }
        }
        self.update(img, contour, self.num_levels, self.distances, self.angles)


    def update(self, img:np.array, contour:np.array, num_levels:int, distances: list, angles: list) -> None:
        self._img = img
        self._contour = contour

        # Cropp image
        self._cropped = cropInterRect(self.img, self.contour)
        # Rescale
        self._cropped = np.interp(self._cropped, (np.min(self._cropped), np.max(self._cropped)), (0, num_levels - 1)).astype(int)
        # GLCM
        self._glcms = graycomatrix(self.cropped, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)



    @property
    def glcm_settings(self): return self._glcm_settings
    @property
    def num_levels(self): return self._glcm_settings['num_levels']
    @property
    def distances_name(self): return list(self._glcm_settings['distances'].keys())
    @property
    def angles_name(self): return list(self._glcm_settings['angles'].keys())
    @property
    def distances(self): return list(self._glcm_settings['distances'].values())
    @property
    def angles(self): return list(self._glcm_settings['angles'].values())
    @property
    def glcm_coef(self): return self._glcm_coef
    @property
    def img(self): return self._img
    @property
    def contour(self): return self._contour
    @property
    def cropped(self): return self._cropped
    @property
    def glcms(self): return self._glcms


    # @property
    # def energies(self): return graycoprops(self.glcms, 'energy')
    # @property
    # def contrasts(self): return graycoprops(self.glcms, 'contrast')
    # @property
    # def correlations(self): graycoprops(self.glcms, 'correlation')
    # @property
    # def variances(self): graycoprops(self.glcms, 'variance')
    # Inverse Difference Moment


    @staticmethod
    def calc_coef(glcm: np.array) -> dict:
        ''' pre-calculate coefficients of Haralick Texture
            Args:
                glcm: glcm, 2d, symmetric, normalized
            Returns:
                coefs: coefficients
        '''
        N_g = glcm.shape[0]
        eps = np.spacing(1) # epsilon, for log calculation
        cols, rows = np.meshgrid(np.arange(1,glcm.shape[1]+1), np.arange(1,glcm.shape[0]+1))
        col_mean = np.sum(cols * glcm)
        row_mean = np.sum(rows * glcm)
        col_std = np.sqrt(np.sum((cols - col_mean) ** 2 * glcm))
        row_std = np.sqrt(np.sum((rows - row_mean) ** 2 * glcm))
        p_x = np.sum(glcm, axis=1, keepdims=True)
        p_y = np.sum(glcm, axis=0, keepdims=True)

        # p_{x+y} from 2 to 2*N_g
        p_xplusy = np.zeros(2*N_g+1)
        for i in range(1,N_g+1):
            for j in range(1,N_g+1):
                p_xplusy[i+j] += glcm[j-1][i-1]
        p_xplusy = p_xplusy[2:]

        ## p_{x-y} from 0 to N_g-1
        p_xminusy = np.zeros(N_g)
        for i in range(1,N_g+1):
            for j in range(1,N_g+1):
                p_xminusy[abs(i-j)] += glcm[j-1][i-1]

        return {
            'N_g':N_g,
            'eps': eps,
            'cols':cols,
            'rows':rows,
            'col_mean':col_mean,
            'row_mean':row_mean, 
            'col_std': col_std,
            'row_std': row_std,
            'p_x':p_x,
            'p_y':p_y,
            'p_xplusy':p_xplusy,
            'p_xminusy': p_xminusy
        }
    

    @staticmethod
    def getFeatures(glcm, glcm_prefix=''):
        ''' 這版本主要以 實驗室 matlab 版 radiomics 為主
        - log 以 log 為主(e-based), entropy 除外(2-based)
        - 有些除0和log0用 epsilon 防止錯誤，參考 pyradiomics
        '''
        features = [
            'energy', 'contrast', 'correlation', 'variance', 'entropy',
            'sum_average', 'sum_variance', 'sum_entropy', 'difference_variance', 'difference_entropy',
            'info_measure_corr_1', 'info_measure_corr_2', 'max_corr_coeff',
            'dissimilartiy', 'autocorrelation', 'cluster_shade', 'cluster_prominence', 'maximum_probability',
            'inverse_difference', 'inverse_difference_norm', 'inverse_difference_moment', 'inverse_difference_moment_norm'
        ]
        if glcm_prefix != '': 
            for idx, f in enumerate(features): features[idx] = f"{glcm_prefix}_{f}"
        
        coefs = HaralickTextureFeaturesExtractor.calc_coef(glcm)
        
        # Energy
        energy = np.sum(glcm**2)

        # Contrast
        contrast = np.sum((np.abs(coefs['rows'] - coefs['cols']) ** 2) * glcm)

        # Correlation
        covariance = np.sum((coefs['rows'] - coefs['row_mean']) * (coefs['cols'] - coefs['col_mean']) * glcm)
        if coefs['row_std'] * coefs['col_std'] == 0: correlation = 1  # Set elements that would be divided by 0 to 1.
        else: correlation = covariance / (coefs['row_std'] * coefs['col_std'])
        

        # Variance
        variance = np.sum(((coefs['rows'] - np.mean(glcm)) ** 2) * glcm)

        # Entropy
        entropy = -np.sum(glcm*np.log2(glcm+coefs['eps']))

        # Sum Average ## from 2 to 2N_g
        sum_average = np.sum(  (np.ogrid[0:len(coefs['p_xplusy'])]+2) * coefs['p_xplusy'])

        # Sum Entropy ## log(0) is not defined
        sum_entropy = -np.sum(coefs['p_xplusy'] * np.log(coefs['p_xplusy']+coefs['eps']))

        # Sum Variance ## from 2 to 2N_g
        sum_variance = np.sum( (((np.ogrid[0:len(coefs['p_xplusy'])]+2)-sum_entropy)**2) * coefs['p_xplusy'] )

        # Difference Variance
        difference_variance = np.sum( (np.ogrid[0:coefs['N_g']] - np.mean(coefs['p_xminusy']))**2 * coefs['p_xminusy'] )

        # Difference Entropy
        difference_entropy = -np.sum(coefs['p_xminusy'] * np.log(coefs['p_xminusy']+coefs['eps']))

        # Information Measures of Correlation I and II
        HXY1 = -np.sum( glcm*np.log2(coefs['p_x']*coefs['p_y']+coefs['eps']) )
        HXY2 = -np.sum( coefs['p_x']*coefs['p_y']*np.log2(coefs['p_x']*coefs['p_y']+coefs['eps']) )
        HX = -np.sum(coefs['p_x']*np.log2(coefs['p_x']+coefs['eps']))
        HY = -np.sum(coefs['p_y']*np.log2(coefs['p_y']+coefs['eps']))
        info_measure_corr_1 = (entropy-HXY1)/max(HX, HY)
        info_measure_corr_2 = math.sqrt(1-math.exp(-2*(HXY2-entropy)))

        # Maximum Correlation Coefficient
        ## 這邊和原版寫法不太一樣
        Q = np.sum((glcm[:, np.newaxis, :] * glcm[np.newaxis, :, :]) / (coefs['p_x'].flatten()[:, np.newaxis, np.newaxis] * coefs['p_y'].flatten() + coefs['eps']), axis=2)
        # display(Q)
        Q_eigenValue = np.sort( np.linalg.eigvals(Q) )
        if len(Q_eigenValue) < 2: max_corr_coeff = 1
        else:
            second_largest_eigenvalue = Q_eigenValue[-2]
            max_corr_coeff = np.sqrt(second_largest_eigenvalue)

        # Dissimilarity
        dissimilartiy = np.sum( np.abs(coefs['rows']-coefs['cols'])*glcm ) # = np.sum( np.ogrid[0:len(coefs['p_xminusy'])]*coefs['p_xminusy'] )     

        # Autocorrelation
        autocorrelation = np.sum( coefs['rows']*coefs['cols']*glcm )      

        # Cluster Shade
        cluster_shade = np.sum( (coefs['rows']+coefs['cols']-coefs['row_mean']-coefs['col_mean'])**3 * glcm )       

        # Cluster Prominence
        cluster_prominence = np.sum( (coefs['rows']+coefs['cols']-coefs['row_mean']-coefs['col_mean'])**4 * glcm )      

        # Maximum probability
        maximum_probability = np.amax(glcm)     

        # Inverse difference
        inverse_difference = np.sum( glcm / (1+np.abs(coefs['cols']-coefs['rows'])) ) # = np.sum( coefs['p_xminusy'] / (1+np.ogrid[0:len(coefs['p_xminusy'])]) )        

        # Inverse difference normalized
        inverse_difference_norm = np.sum( glcm / (1+np.abs(coefs['cols']-coefs['rows'])/coefs['N_g']) ) # = np.sum( coefs['p_xminusy'] / (1+np.ogrid[0:len(coefs['p_xminusy'])] / coefs['N_g']) )     

        # Inverse Difference Moment
        inverse_difference_moment = np.sum(glcm / (1 + (coefs['rows'] - coefs['cols']) ** 2))  # = np.sum( coefs['p_xminusy'] / (1+np.ogrid[0:len(coefs['p_xminusy'])]**2) )        

        # Inverse Difference Moment normalized
        inverse_difference_moment_norm = np.sum( glcm / (1+(coefs['cols']-coefs['rows'])**2/coefs['N_g']**2) ) # = np.sum( coefs['p_xminusy'] / (1+(np.ogrid[0:len(coefs['p_xminusy'])]**2 / coefs['N_g']**2)) )
        
        return pd.DataFrame(
            [[
                energy, contrast, correlation, variance, entropy,
                sum_average, sum_variance, sum_entropy, difference_variance, difference_entropy,
                info_measure_corr_1, info_measure_corr_2, max_corr_coeff,
                dissimilartiy, autocorrelation, cluster_shade, cluster_prominence, maximum_probability,
                inverse_difference, inverse_difference_norm, inverse_difference_moment, inverse_difference_moment_norm
            ]],
            columns=features
        )
    

    def all(self, feature_prefix=''):
        features = list()
        for idx_d, dis in enumerate(self.distances_name):
            for idx_a, ang in enumerate(self.angles_name):
                features.append( self.getFeatures(self.glcms[:,:,idx_d, idx_a], f'{feature_prefix}_d{dis}a{ang}') )

        features = pd.concat(features, axis=1)

        return features
    
    
    # def getEnergy(self, glcm):
    #     ''' ...
    #         Args:
    #             glcm: glcm, 2d, symmetric, normalized
    #         Returns:
    #             energy: energy
    #     '''
    #     return np.sum(glcm**2)
    

    # def getConrast(glcm):
    #     ''' ...
    #         Args:
    #             glcm: glcm, 2d, symmetric, normalized
    #         Returns:
    #             contrast: contrast
    #     '''
    #     return np.sum((np.abs(rows - cols) ** 2) * glcm)
    

    # def getConrast(glcm):
    #     ''' ...
    #         Args:
    #             glcm: glcm, 2d, symmetric, normalized
    #         Returns:
    #             contrast: contrast
    #     '''
    #     return np.sum((np.abs(rows - cols) ** 2) * glcm)