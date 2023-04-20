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
path_las = r"D:\Baumartenklassifizierung\data\train_downsampled"

#%%

def augment(path_las, rotate_h_max = 22.5, rotate_v_max = 180,
            translate_max = 0.25, sampling_max = 0.1):
    
    """
    Parameters
    ----------
    path_las : str
        Path to the input las file.
    rotate_h_max : float, optional
        Maximum random rotation along the horizontal axes in degree. The
        default is 22.5.
    rotate_v_max : float, optional
        Maximum random rotation along the vertical axis in degree. The default
        is 180.
    translate_max : float, optional
        Maximum random translation relative to the maximum expansion from the
        center as a fraction. The default is 0.25.
    sampling_max : float, optional
        Maximum downsampling as a fraction. The default is 0.1.

    Returns
    -------
    XYZ point coordinates in np.array.
    """
    
    # read in las file
    las = lp.read(path_las)
    
    # turn coordinates into numpy array
    points = np.stack((las.X, las.Y, las.Z), axis = 1)
    points = points * las.header.scale
    
    # sub-sampling
    s_num = int((1 - sampling_max) * points.shape[0])
    s_idx = np.random.choice(np.arange(points.shape[0]), s_num, replace = False)
    points = points[s_idx,:]
    
    # prepare transformation matrix
    transform = np.identity(3, float)
    
    # x rotation matrix
    r_x_rad = np.radians(np.random.uniform(-rotate_h_max, rotate_h_max))
    r_x_mat = np.array([
        [1,               0,                0],
        [0, np.cos(r_x_rad), -np.sin(r_x_rad)],
        [0, np.sin(r_x_rad),  np.cos(r_x_rad)]])
    
    # y rotation matrix
    r_y_rad = np.radians(np.random.uniform(-rotate_h_max, rotate_h_max))
    r_y_mat = np.array([
        [ np.cos(r_y_rad), 0, np.sin(r_y_rad)],
        [               0, 1,               0],
        [-np.sin(r_y_rad), 0, np.cos(r_y_rad)]])
    
    # z rotation matrix
    r_z_rad = np.radians(np.random.uniform(-rotate_v_max, rotate_v_max))
    r_z_mat = np.array([
        [np.cos(r_z_rad), -np.sin(r_z_rad), 0],
        [np.sin(r_z_rad),  np.cos(r_z_rad), 0],
        [              0,                0, 1]])
    
    # combine rotation matrices
    r_xyz = np.dot(np.dot(r_z_mat, r_y_mat), r_x_mat)
    transform[:3,:3] = r_xyz
    
    # apply transformation
    points = np.matmul(points, transform.T)
    
    # # create a new las file
    # new_header = lp.LasHeader(point_format = 0, version = "1.2")
    # new_header.offsets = las.header.offset
    # new_header.scales = las.header.scale
    # new_las = lp.LasData(new_header)
    # new_las.x = points[:,0]
    # new_las.y = points[:,1]
    # new_las.z = points[:,2]
        
    # # write augmented las
    # path_out_full = os.path.join(path_out, os.path.basename(path_las))
    # new_las.write(path_out_full)
        
    # return path
    return points

#%%

# # execution for all training las files
# for path_curr in glob.glob(os.path.join(path_las, "*.las")):
#     augment(path_curr)

# execution for a single file
pts = augment(r"D:\Baumartenklassifizierung\data\train_downsampled\03498.las")
print(pts.shape)
