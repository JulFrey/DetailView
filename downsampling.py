# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
import os
import glob
import laspy as lp
import numpy as np

# set paths & variables
path_las = r"S:\3D4EcoTec\train"
path_out = r"D:\Baumartenklassifizierung\data\raw"

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
    
    # read in las file
    las = lp.read(path_las)
    
    # turn coordinates into numpy array
    points = np.stack((las.X, las.Y, las.Z), axis = 1)
    points = points * las.header.scale
    
    # check the number of points
    if points.shape[0] < min_n:
        return ""
    
    # calculate minimum coordinate values
    min_coords = np.min(points, axis = 0)
    
    # calculate voxel indices for each point
    voxel_indices = np.floor((points - min_coords) / res_pc).astype(int)
    
    # collect non-empty voxels
    unique_voxel_indices = np.unique(voxel_indices, axis = 0)
    points = (unique_voxel_indices + 0.5) * res_pc + min_coords
    
    # # calculate average point per voxel (takes too long)
    # voxel_uniques = np.unique(voxel_indices, axis = 0)
    # new_points = np.zeros(voxel_uniques.shape)
    # for idx in range(voxel_uniques.shape[0]):
    #     new_points[idx,:] = np.mean(points[(voxel_indices == voxel_uniques[idx,:]).all(axis = 1),:], axis = 0)
    # points = new_points
    # del new_points
    
    # create a new las file
    new_header = lp.LasHeader(point_format = 0, version = "1.2")
    new_header.offsets = las.header.offset
    new_header.scales = las.header.scale
    new_las = lp.LasData(new_header)
    new_las.x = points[:,0]
    new_las.y = points[:,1]
    new_las.z = points[:,2]
    
    # write downsampled las
    path_out_full = os.path.join(path_out, os.path.basename(path_las))
    new_las.write(path_out_full)
    
    # return path
    return path_out_full

#%%

# # execution for all training las files
# for path_curr in glob.glob(os.path.join(path_las, "*.las")):
#     downsample(path_curr, path_out)

# execution for a single file
downsample(r"D:\Baumartenklassifizierung\data\raw\03498.las", path_out)
