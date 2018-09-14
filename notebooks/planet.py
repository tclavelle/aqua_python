"""
Deep Learning for Aquaculture Detection
Planet Imagery Processing Functions

Written by Tyler Clavelle
"""

import rasterio
from rasterio import windows
import skimage
import skimage.io as skio
from skimage import exposure
import os
import sys
import pathlib
import math
import itertools
import functools
import numpy as np
import pandas as pd
from rasterio.plot import show
from osgeo import gdal

####################################################################
# Helper functions
####################################################################

# Get absolute file paths. Returns generator object
def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

# Normalize array
def normalize(arr, arr_max = None):
    ''' Function to normalize an input array to 0-1 '''
    if not arr_max:
        arr_max = arr.max()
        out = arr / arr_max
    else:
        out = arr / arr_max
    return arr / arr_max

# Reorder Planet scenes to RGB
def reorder_to_rgb(image):
    '''reorders planet bands to red, green, blue for imshow'''
    blue = normalize(image[:,:,0])
    green = normalize(image[:,:,1])
    red = normalize(image[:,:,2])
    return np.stack([red, green, blue], axis=-1)

# Reorder Planet scenes to RGB for RASTERIO read images (C,H,W) 
def rasterio_to_rgb(image):
    '''reorders planet bands to red, green, blue for imshow'''
    blue = image[0,:,:]
    green = image[1,:,:]
    red = image[2,:,:]
    return np.stack([red, green, blue], axis=0)

# Contrast stretching algorithm for multiband images
def contrast_stretch_mb(img):
    # Loop over bands of images with shape H,W,C
    for b in range(0, img.shape[2]):
        p2, p98 = np.percentile(img[:,:,b], (2, 98))
        img_scaled = exposure.rescale_intensity(img, in_range=(p2, p98))
        img[:,:,b] = img_scaled[:,:,b]
    return img
        
    
    