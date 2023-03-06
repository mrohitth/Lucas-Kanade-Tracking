import numpy as np
import numpy as np
from LucasKanadeAffine import *
from InverseCompositionAffine import *
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline
import cv2
from scipy.ndimage import affine_transform
from scipy.ndimage.morphology import binary_dilation, binary_erosion

"""
:param image1: Images at time t
:param image2: Images at time t+1
:param threshold: used for LucasKanadeAffine
:param num_iters: used for LucasKanadeAffine
:param tolerance: binary threshold of intensity difference when computing the mask
:return: mask: [nxm]
"""

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    
    mask = np.zeros(image1.shape, dtype=bool)
    '''
    compostition affine
    '''
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    '''
    inverse compostition affine
    '''
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    
    image2_warp=cv2.warpAffine(image2,M[:2],image1.T.shape)
    
    image2_erode    = binary_erosion(image2_warp)
    image2_dilation = binary_dilation(image2_erode)
    
    diff = np.abs(image1-image2_dilation)
    
    mask = (diff>tolerance)
    
    # print(mask)
    return mask