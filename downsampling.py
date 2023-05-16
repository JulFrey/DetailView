# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
import os
import glob
import pdal

# set paths & variables
path_las = r"S:\3D4EcoTec\train"
path_out = r"C:\TLS\down\train"

# create output folder
if not os.path.exists(path_out):
   os.makedirs(path_out)

#%%

def downsample(path_las, path_out, res_pc = 0.01, min_n = 100):
    
    """
    Parameters
    ----------
    path_las : str
        Path to the input las file.
    path_out : str
        Path to the output folder.
    res_pc : float, optional
        Target las resolution. The default is 0.01.
    min_n : float, optional
        Minimum number of points. The default is 100.
    
    Returns
    -------
    Path to new point cloud.
    """

    # get output path
    path_out_full = os.path.join(path_out, os.path.basename(path_las))
    
    # setting up downsample
    las_reader = pdal.Reader(path_las)
    las_filter = pdal.Filter(type = "filters.voxelcentroidnearestneighbor", cell = res_pc)
    las_writer = pdal.Writer(path_out_full, dataformat_id = 0)
    
    # get number of points
    pipeline = pdal.Pipeline([las_reader])
    pipeline.execute()
    n = pipeline.metadata["metadata"]['readers.las']['count']
    
    # skip if too few points
    if n < min_n:
        return ""
    
    # downsample point cloud
    pipeline = las_reader | las_filter | las_writer
    pipeline.execute()
    
    # return path
    return path_out_full

#%%

# execution for all training las files
for path_curr in glob.glob(os.path.join(path_las, "*.las")):
    downsample(path_curr, path_out)

# # execution for a single file
# downsample(r"D:\Baumartenklassifizierung\data\12781.las", path_out)
