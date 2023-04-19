# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
import os
import math
import random
import laspy as lp
import numpy as np

# set paths & variables
path_las = r"D:\Baumartenklassifizierung\data\raw"
path_out = r"D:\Baumartenklassifizierung\data\processed"

# create output folder
if not os.path.exists(path_out):
   os.makedirs(path_out)

#%%

def augment(path_las, path_out, rotate_h_max = 22.5, rotate_v_max = 180,
            translate_max = 0.25, scale_max = 0.25):
    
    """
    Parameters
    ----------
    path_las : str
        Path to the input las file.
    path_out : str
        Path to the output folder.
    rotate_h_max : float, optional
        Maximum random rotation along the horizontal axes in degree. The
        default is 22.5.
    rotate_v_max : float, optional
        Maximum random rotation along the vertical axis in degree. The default
        is 180.
    translate_max : float, optional
        Maximum random translation relative to the maximum expansion from the
        center as a fraction. The default is 0.25.
    scale_max : float, optional
        Maximum random scaling as a fraction. The default is 0.25.

    Returns
    -------
    Path to new point cloud.
    """
    
    # read in las file
    las = lp.read(path_las)
    
    # turn coordinates into numpy array
    points = np.stack((las.X, las.Y, las.Z), axis = 1)
    points = points * las.header.scale
    
    # center data
    # (bedingt sinnvoll, weil die Wolke durch das drehen auch verschoben wird)
    points = points - np.median(points, axis = 0)
    
    # prepare transformation matrix
    transform = np.identity(4, float)
    
    # x rotation matrix
    r_x_rad = math.radians(random.uniform(-rotate_h_max, rotate_h_max))
    r_x_mat = np.array([
        [1,                 0,                  0],
        [0, math.cos(r_x_rad), -math.sin(r_x_rad)],
        [0, math.sin(r_x_rad),  math.cos(r_x_rad)]])
    
    # y rotation matrix
    r_y_rad = math.radians(random.uniform(-rotate_h_max, rotate_h_max))
    r_y_mat = np.array([
        [ math.cos(r_y_rad), 0, math.sin(r_y_rad)],
        [                 0, 1,                 0],
        [-math.sin(r_y_rad), 0, math.cos(r_y_rad)]])
    
    # z rotation matrix
    r_z_rad = math.radians(random.uniform(-rotate_v_max, rotate_v_max))
    r_z_mat = np.array([
        [math.cos(r_z_rad), -math.sin(r_z_rad), 0],
        [math.sin(r_z_rad),  math.cos(r_z_rad), 0],
        [                0,                  0, 1]])
    
    # combine rotation matrices
    r_xyz = np.dot(np.dot(r_z_mat, r_y_mat), r_x_mat)
    transform[:3,:3] = r_xyz
    
    # translate
    # (bedingt sinnvoll, weil die Wolke durch das drehen auch verschoben wird)
    # (vielleicht erst drehen, dann zentrieren, dann verschieben?
    # -> trennen von drehen & skalieren?)
    t_max = max(np.max(abs(points), axis = 0)) * translate_max
    t_xyz = np.random.uniform(-t_max, t_max, 3)
    transform[0:3,-1] = t_xyz
    
    # scale
    s_val = (1 + random.uniform(-scale_max, scale_max))
    s_xyz = np.diag([s_val, s_val, s_val, 1])
    transform = np.dot(transform, s_xyz)
    
    # apply transformation
    points = np.append(points, np.ones((points.shape[0],1), float), axis = 1)
    points = np.matmul(points, transform.T)
    points = points[:,:3]
    
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
#     augment(path_curr, path_out)

# execution for a single file
augment(r"D:\Baumartenklassifizierung\data\raw\03498.las", path_out)
