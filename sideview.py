# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd

def points_to_images(points, res_im = 256, num_side = 4, plot = False,
                     max_n = 500000, debug = False):
    
    """
    Parameters
    ----------
    points : np.array
        XYZ point coordinates in np.array.
    res_im : int, optional
        Edge length of the quadratic tensor. The default is 256.
    num_side : int, optional
        number of side views. The default is 4.
    plot : bool, optional
        Plot the results for debugging. The default is False.
    max_n: int, optional
        Maximum number of points to be used for side- & topviews. The default
        is 500,000.

    Returns
    -------
    views : np.array
        Different 2D views stacked in np.array.
    """
    
    # # read in las file
    # las = rl.read_las(path_las)
    
    # prepare view array
    views = np.zeros((num_side + 3, res_im, res_im), dtype = "float32")
    
    # add DBH section view
    # TODO: remove background points from section view
    if debug: print('create dbh section view')
    views[num_side + 2,:,:] = sectionview(points, res_im = res_im, plot = plot, debug = debug)
    
    # subsample points to max_n points
    if points.shape[0] > max_n:
        points = points[np.random.choice(np.arange(points.shape[0]), max_n, replace = False),:]
    
    # center point cloud
    if debug: print('center point cloud')
    points = points - np.median(points, axis = 0)
    
    # scale point cloud using the maximum axis
    if debug: print('scale point cloud')
    points = points / np.max(np.abs(points))
    
    # add top view
    if debug: print('create top view')
    views[0,:,:] = topview(points, res_im = res_im, plot = plot)
    
    # loop through perspectives
    deg_steps = np.arange(0, 360, 360/num_side)
    # deg_steps = np.linspace(0, 180, num = num_side)
    for i in range(num_side):
        if debug: print('sideview: ' + str(i))
        
        # get required rotation
        deg = deg_steps[i]
        
        # z rotation matrix
        rad = np.radians(deg)
        rot = np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad),  np.cos(rad), 0],
            [          0,            0, 1]])
        
        # rotate point cloud
        points_rot = np.matmul(points, rot.T)
        
        # get side images
        views[i+1,:,:] = sideview(points_rot, res_im = res_im, plot = plot)
    
    # add bottom view
    if debug: print('bottom view')
    views[num_side + 1,:,:] = topview(points, res_im = res_im, inverse = True, plot = plot)
    
    # return
    return views

# creating topview
def topview(points, res_im = 256, inverse = False, plot = False):
    
    """
    Parameters
    ----------
    points : np.array
        XYZ point coordinates in np.array.
    res_im : int, optional
        Edge length of the quadratic tensor. The default is 256.
    inverse : bool, optional
        Calculate bottom view instead. The default is False.
    plot : bool, optional
        Plot the results for debugging. The default is False.

    Returns
    -------
    top_image : np.array
        2D view in np.array.
    """
    
    # find the minimum and maximum values of the x, y, and z coordinates
    x_min, y_min = np.min(points[:,[0,1]], axis = 0)
    x_max, y_max = np.max(points[:,[0,1]], axis = 0)
    x_med, y_med = np.median(points[:,[0,1]], axis = 0)
    
    # create an empty numpy array to store the depth image
    top_image = np.ones((res_im, res_im)) * -999
    
    # calculate the size of each pixel in the x and y dimensions
    size = max(x_max - x_min, y_max - y_min) / res_im
    
    # determine longer axis
    max_axis_x = (x_max - x_min) > (y_max - y_min)
    
    # calculate image coordinates
    if inverse:
        if max_axis_x:
            x_pos = np.array((points[:,0] - x_min) / size, dtype = int)
            y_pos = np.array((points[:,1] - y_med) / size, dtype = int) + int(res_im/2)
        else:
            x_pos = np.array((points[:,0] - x_med) / size, dtype = int) + int(res_im/2)
            y_pos = np.array((points[:,1] - y_min) / size, dtype = int)
    else:
        if max_axis_x:
            x_pos = np.array((points[:,0] - x_min) / size, dtype = int)
            y_pos = np.array((points[:,1] - y_med) / size, dtype = int) + int(res_im/2)
        else:
            x_pos = np.array((points[:,0] - x_med) / size, dtype = int) + int(res_im/2)
            y_pos = np.array((points[:,1] - y_min) / size, dtype = int)
    
    # save as pandas array
    # wrap indices to avoid out of range indexing
    points = pd.DataFrame({
        "x": x_pos % res_im,
        "y": y_pos % res_im,
        "depth": points[:,2]})
    
    # get minimum/maximum depth at each unique coordinate
    if inverse:
        points = points.groupby(["x", "y"])["depth"].min().reset_index()
    else:
        points = points.groupby(["x", "y"])["depth"].max().reset_index()
    
    # overwrite depth values
    top_image[points["x"], points["y"]] = points["depth"]
    
    # replace dummy values with value
    top_image[top_image == -999] = 0
    
    # show image
    if plot:
        plt.imshow(top_image, interpolation = 'nearest')
        plt.show()
    
    # return array
    return top_image

# creating sideview
def sideview(points, res_im = 256, plot = False):
    
    """
    Parameters
    ----------
    points : np.array
        XYZ point coordinates in np.array.
    res_im : int, optional
        Edge length of the quadratic tensor. The default is 256.
    plot : bool, optional
        Plot the results for debugging. The default is False.

    Returns
    -------
    side_image : np.array
        2D view in np.array.
    """
    
    # find the minimum and maximum values of the x, y, and z coordinates
    x_min, z_min = np.min(points[:,[0,2]], axis = 0)
    x_max, z_max = np.max(points[:,[0,2]], axis = 0)
    x_med, z_med = np.median(points[:,[0,2]], axis = 0)
    
    # create an empty numpy array to store the depth image
    side_image = np.ones((res_im, res_im)) * -999
    
    # calculate the size of each pixel in the x and y dimensions
    size = max(x_max - x_min, z_max - z_min) / res_im
    
    # determine longer axis
    max_axis_x = (x_max - x_min) > (z_max - z_min)
    
    # calculate image coordinates
    if max_axis_x:
        x_pos = np.array((points[:,0] - x_min) / size, dtype = int)
        z_pos = np.array((points[:,2] - z_med) / size, dtype = int) + int(res_im/2)
    else:
        x_pos = np.array((points[:,0] - x_med) / size, dtype = int) + int(res_im/2)
        z_pos = np.array((points[:,2] - z_min) / size, dtype = int)
    
    # save as pandas array
    # wrap indices to avoid out of range indexing
    points = pd.DataFrame({
        "x": x_pos % res_im,
        "z": z_pos % res_im,
        "depth": points[:,1]})
    
    # get maximum depth at each unique coordinate
    points = points.groupby(["x", "z"])["depth"].max().reset_index()
    
    # overwrite depth values
    side_image[points["x"], points["z"]] = points["depth"]
    
    # replace dummy values with value
    side_image[side_image == -999] = 0
    
    # show image
    if plot:
        plt.imshow(side_image, interpolation = 'nearest')
        plt.show()
    
    # return array
    return side_image


# creating sideview
def sectionview(points, res_im = 256, plot = False, debug = False):
    
    """
    Parameters
    ----------
    points : np.array
        XYZ point coordinates in np.array.
    res_im : int, optional
        Edge length of the quadratic tensor. The default is 256.
    plot : bool, optional
        Plot the results for debugging. The default is False.

    Returns
    -------
    section_image : np.array
        2D view in np.array.
    """
    
    # extract DBH section
    section = points[(points[:,2] < 1.5) & (points[:,2] > 1),:]
    if debug: print('n points in section section: ' + str(section.shape[0]))
    
    # skip if section nearly empty
    if section.shape[0] > 50:
    
        # set up the clustering algorithm
        dbscan = DBSCAN(eps = 0.10, min_samples = 10)
        
        # fit the clustering algorithm
        dbscan.fit(section)
        
        # get largest cluster
        labels = dbscan.labels_
        
        # catch if no cluster > 10 points is found
        if not (np.array(labels) == -1).all():
            largest_cluster_label = np.argmax(np.bincount(labels[labels!=-1]))
            largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
            section = section[largest_cluster_indices]
    
        # center point cloud
        section = section - np.median(section, axis = 0)
        
        # scale point cloud
        section = section / np.max(np.abs(section))
        
        # create sideview
        section_image = sideview(section, res_im = res_im, plot = plot)
        
    else:
        # create empty array
        section_image = np.zeros((res_im, res_im))
    
    # return array
    return section_image
