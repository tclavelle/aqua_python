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
        
#################################################################################################
# Functions
#################################################################################################

# Process a single PlanetScope 3B GeoTiff into
def planet2chips(tiff_directory, chip_directory, chip_size = 512):
    
    """ Creates image chips (GeoTiffs and PNGs) of a GeoTiff file in a 
    specified directory and saves in new directory location 
    """
  
    # Get all analytic SR GeoTiff filnames in specified directory
    files = np.array(os.listdir(tiff_directory))
    tiff = pd.Series(files).str.contains('SR.tif')
    file = files[tiff][0]

    # Get image name to use for creating directory
    image_name = file.split("_")[0:3]
    image_name = "%s_%s_%s" % (image_name[0], image_name[1], image_name[2])

    # Image chip destination directory and subdirectories
    image_dir = os.path.join(chip_directory, image_name)   

    chip_dir = os.path.join(image_dir,'chips')
    png_dir = os.path.join(image_dir, 'pngs')

    # Print filenames
    print('filename: ' + file + '\n' + 'image name: ' + image_name)

    # Make directories to store raw and rgb image chips
    pathlib.Path(chip_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(png_dir).mkdir(parents=True, exist_ok=True)
    
    # Iterate over image blocks as defined by chip_size and save new GeoTiffs
    with rasterio.open(os.path.join(tiff_directory, file)) as src:
                
        # Read full src image and calculate percentiles for contrast stretchin
        full_src = src.read()
        print(full_src.shape)
        
        # Create windows of desired size
        rows1 = np.arange(0,full_src.shape[1], chip_size)
        rows2 = np.arange(chip_size,full_src.shape[1], chip_size)
        
        cols1 = np.arange(0,full_src.shape[2], chip_size)
        cols2 = np.arange(chip_size,full_src.shape[2], chip_size)
        
        # arrange into tuples
        rows = list(zip(rows1, rows2))
        cols = list(zip(cols1, cols2))
        
        # Arrange into tuples of windows to read
        windows = [ (a,b) for a in rows for b in cols ]                        
        
        # Get block dimensions of src
        for window in windows:

            r = src.read((1,2,3,4), window=window)

            if 0 in r:
                continue

            else:
                
                # Get start row and column for file name
                rmin = window[0][0]
                cmin = window[1][0]
            
                # Scale variable. Note bands of Planet imagery go BGR
                b = src.read((3,2,1), window=window)
                # Swap axis from rasterio order (C,H,W) to order expected by skio (H,W,C)
                b = np.moveaxis(b, 0, 2)
                b = contrast_stretch_mb(b)
                png_file = png_dir + '/' + image_name + '_' + str(rmin) + '_' + str(cmin) + '.png'
                skio.imsave(png_file, b)                

                # Open a new GeoTiff data file in which to save the raw image chip
                with rasterio.open((chip_dir + '/' + image_name + '_' + str(rmin) + '_' + str(cmin) + '.tif'), 'w', driver='GTiff',
                           height=r.shape[1], width=r.shape[2], count=4,
                           dtype=rasterio.uint16, crs=src.crs, 
                           transform=src.transform) as new_img:

                    # Write the raw image to the new GeoTiff
                    new_img.write(r)
                    

# Process images in each planet order folder
def process_planet_orders(source_dir, target_dir):
    
    """Find unique PlanetScope scenes in a directory of Planet order folders
    and process newly added scenes into image chips"""
    
    # Get list of all planet orders in source directory
    orders = np.array(next(os.walk(source_dir))[1])
    # Add full path to each order directory
    orders = [os.path.join(source_dir, o) for o in orders]
    
    scenes = []
    scene_paths = []
    
    for o in orders:
        # scenes in order
        s_ids = np.array(next(os.walk(o))[1])
        s_ids_paths = [os.path.join(source_dir,o,s) for s in s_ids]
        
        # add to lists
        scenes.append(s_ids)
        scene_paths.append(s_ids_paths)
    
    # Flatten lists
    scenes = list(np.concatenate(scenes))
    print(len(scenes))
    scene_paths = list(np.concatenate(scene_paths))
    
    # Check which scenes already have chip folders
    scenes_exist = np.array(next(os.walk(target_dir))[1])
    
    scenes_to_process = []
    scene_paths_to_process = []
    
    # Remove scenes that already exist from list of scenes to process
    for s, sp in zip(scenes, scene_paths):
        if s not in scenes_exist:
            scenes_to_process.append(s)
            scene_paths_to_process.append(sp)            


    # Apply GeoTiff chipping function to each unprocessed scene
    for sp in scene_paths_to_process:
        # print(sp)
        try:
            planet2chips(tiff_directory = sp, chip_directory = target_dir, chip_size = 512) 
        except IndexError:
            print('Scene is not an AnalyticMS_SR tiff file')
            continue


# Function to copy the tiffs of PNGs selected for labeling and make directories for each chip
def copy_chip_tiffs(label_dir, chips_dir, prepped_dir):
    
    """ Take a VGG label JSON file and, for each labeled image, create a directory in 
    prepped_planet containing the raw tiff and a directory of class masks
    """
    # Read annotations
    pngs = os.listdir(label_dir)
    pngs = [png for png in pngs if png != '.DS_Store'] # remove stupid DS_Store file
    
    # Extract filenames and drop .png extension
    chips = [c.split('.png')[0] for c in pngs]
    
    # Loop over chips
    for chip in chips:
        
        # Make directory for chip in prepped dir
        chip_dir = os.path.join(prepped_dir, chip)
        # Create "image" dir for tiff image
        image_dir = os.path.join(chip_dir, 'image')
        
        # Make chip directory and subdirectories
        for d in [chip_dir, image_dir]:
            pathlib.Path(d).mkdir(parents=True, exist_ok=True)
                
        # Now locate the tiff file and copy into chip directory
        # Get scene name for chip
        scene = chip.split('_')[0:3]
        scene = "%s_%s_%s" % (scene[0], scene[1], scene[2])
        
        # Locate and copy tiff file
        tiff = os.path.join(chips_dir, scene, 'chips', (chip + '.tif'))
        copy2(tiff, image_dir)

# Function to take a JSON of aquaculture labels exported from VGG and produce a single mask per class      
def masks_from_labels(labels, prepped_dir):
    
    # Read annotations
    annotations = json.load(open(labels))
    annotations = list(annotations.values())  # don't need the dict keys
    
    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]
    
    # Loop over chips
    for a in annotations:
        
        # Get chip and directory
        chip = a['filename'].split('.png')[0]                        
        chip_dir = os.path.join(prepped_dir, chip)
        
        # Create a directory to store masks
        masks_dir = os.path.join(chip_dir, 'class_masks')
        pathlib.Path(masks_dir).mkdir(parents=True, exist_ok=True)
        
        # Read geotiff for chip
        gtiff = chip_dir +  '/' + 'image' + '/' + chip + '.tif'
        src = rasterio.open(gtiff)

        # Use try to only extract masks for chips with complete annotations and class labels
        try:

            """Code for processing VGG annotations from Matterport balloon color splash sample"""
            # Load annotations
            # VGG Image Annotator saves each image in the form:
            # { 'filename': '28503151_5b5b7ec140_b.jpg',
            #   'regions': {
            #       '0': {
            #           'region_attributes': {},
            #           'shape_attributes': {
            #               'all_points_x': [...],
            #               'all_points_y': [...],
            #               'name': 'polygon'}},
            #       ... more regions ...
            #   },
            #   'size': 100202
            # } 

            # Get the aquaculture class of each polygon    
            polygon_types = [r['region_attributes'] for r in a['regions']]        

            # Get unique aquaculture classes in annotations
            types = set(val for dic in polygon_types for val in dic.values())            

            for t in types:
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. There are stores in the
                # shape_attributes (see json format above) 

                # Pull out polygons of that type               
                polygons = [r['shape_attributes'] for r in a['regions'] if r['region_attributes']['class'] == t]            

                # Draw mask using height and width of Geotiff
                mask = np.zeros([src.height, src.width], dtype=np.uint8)

                for p in polygons:

                    # Get indexes of pixels inside the polygon and set them to 1
                    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])                    
                    mask[rr, cc] = 1            

                # Open a new GeoTiff data file in which to save the image chip
                with rasterio.open((masks_dir + '/' + chip + '_' + str(t) + '_mask.tif'), 'w', driver='GTiff',
                           height=src.shape[0], width=src.shape[1], count=1,
                           dtype=rasterio.ubyte, crs=src.crs, 
                           transform=src.transform) as new_img:

                    # Write the rescaled image to the new GeoTiff
                    new_img.write(mask.astype('uint8'),1)

        except KeyError:                
            print(chip + ' missing aquaculture class assignment')
            # write chip name to file for double checking
            continue

#################################################################################################
# Command Line
#################################################################################################
            
# Set actions of script if sourced from command line
if __name__ == '__main__':
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PlanetScope 3b scenes into training data')
    
    parser.add_argument("command",
                        metavar = "<command>",
                        help = "'preprocess' or 'labels'")
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        
        # Directories for Planet orders (sdir) and the target directory to save processed chips (tdir)
        sdir = '/Users/Tyler-SFG/Desktop/Box Sync/SFG Centralized Resources/Projects/Aquaculture/Waitt Aquaculture/aqua-mapping/aqua-mapping-data/aqua-images/planet'
        tdir = '/Users/Tyler-SFG/Desktop/Box Sync/SFG Centralized Resources/Projects/Aquaculture/Waitt Aquaculture/aqua-mapping/aqua-mapping-data/aqua-images/planet_chips'

        # Run order processing 
        process_planet_orders(sdir, tdir)