# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
import read_las as rl
import numpy as np
import laspy
from typing import Dict, Optional

def augment(las_input, rotate_h_max=22.5, rotate_v_max=180, sampling_max=0.1):
    
    """
    Parameters
    ----------
    las_input : str or laspy.LasData
        Path to a las file or a laspy.LasData object containing the point cloud data.
    rotate_h_max : float, optional
        Maximum random rotation along the horizontal axes in degree. The
        default is 22.5.
    rotate_v_max : float, optional
        Maximum random rotation along the vertical axis in degree. The default
        is 180.
    sampling_max : float, optional
        Maximum downsampling as a fraction. The default is 0.1.
    tree_id_col : str, optional
        The column name in the las file that contains the tree IDs. The default
        is 'TreeID'.
    tree_id : int, optional

    Returns
    -------
    XYZ point coordinates in np.array.
    """
    ## read in las file
    # points = rl.read_las(path_las)
    # Read points
    if isinstance(las_input, str):
        points = rl.read_las(las_input)
    else:
        points = las_input
    # elif isinstance(las_input, laspy.LasData):
    #     if tree_id is None:
    #         raise ValueError("TreeID must be provided when using a laspy object.")
    #     mask = las_input[tree_id_col] == tree_id
    #     points = np.vstack((las_input.x[mask], las_input.y[mask], las_input.z[mask])).T
    # else:
    #     raise TypeError("las_input must be a file path or laspy.LasData object.")


    # sub-sampling
    s_num = int(points.shape[0] * (1 - sampling_max))
    if 0 < s_num < points.shape[0]:
        idx = np.random.permutation(points.shape[0])[:s_num]
        points = points[idx]
    
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
    # in-place rotation (avoids temp array) and in-place recentring
    np.matmul(points, transform.T, out=points)
    z = points[:, 2]
    z -= z.min()
    
    # return path
    return points

