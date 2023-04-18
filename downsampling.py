# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
import os
import laspy as lp
import numpy as np


def downsample(path_las, path_out, res_pc):
    """
    Parameters
    ----------
    path_las : str
        Path to the input las file.
    path_out : str
        Path to the output folder.
    res_pc : float
        Target las resolution.

    Returns
    -------
    Path to new point cloud.
    """
    
    # read in las file
    las = lp.read(path_las)
    
    # turn coordinates into numpy array
    points = np.stack((las.X, las.Y, las.Z), axis = 1)
    points = points * las.header.scale
    
    # calculate bounding box
    min_coords = np.min(points, axis = 0)
    max_coords = np.max(points, axis = 0)
    bounding_box_size = max_coords - min_coords
    
    # calculate number of voxels
    num_voxels = np.ceil(bounding_box_size / res_pc).astype(int)
    
    # calculate voxel indices for each point
    voxel_indices = np.floor((points - min_coords) / res_pc).astype(int)
    
    # collect non-empty voxels
    unique_voxel_indices = np.unique(voxel_indices, axis = 0)
    points = (unique_voxel_indices + 0.5) * res_pc + min_coords
    
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

downsample(r"S:\3D4EcoTec\train\18146.las", r"D:\Baumartenklassifizierung\data\raw", 0.01)