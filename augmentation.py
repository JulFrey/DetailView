# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
import read_las as rl
import numpy as np

def augment(path_las, rotate_h_max = 22.5, rotate_v_max = 180,
            sampling_max = 0.1):
    
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
    sampling_max : float, optional
        Maximum downsampling as a fraction. The default is 0.1.

    Returns
    -------
    XYZ point coordinates in np.array.
    """
    
    # read in las file
    points = rl.read_las(path_las)
    
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
    
    # recenter bottom to 0m height
    points[:,2] = points[:,2] - np.min(points[:,2])
    
    # return path
    return points
