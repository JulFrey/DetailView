# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
# import os
# import glob
# import read_las as rl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# set paths & variables
path_las = r"D:\Baumartenklassifizierung\data\processed\03498.las"

#%%

def points_to_images(points, res_im = 256, num_side = 4, plot = False, debug = False):
    
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
    
    # TODO: Subsample to max 100k points
    
    # center point cloud
    if debug: print('center point cloud')
    points = points - np.median(points, axis = 0)
    
    # scale point cloud
    if debug: print('scale point cloud')
    points = points / np.max(abs(points))    
    
    # add top view
    if debug: print('create top view')
    views[0,:,:] = topview(points, res_im = res_im, plot = plot)
    
    # loop through perspectives
    deg_steps = np.linspace(0, 180, num = num_side)
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
    x_min, y_min, z_min = np.min(points, axis = 0)
    x_max, y_max, z_max = np.max(points, axis = 0)
    x_med, y_med, z_med = np.median(points, axis = 0)
    
    # create an empty numpy array to store the depth image
    top_image = np.ones((res_im, res_im)) * -999
    
    # calculate the size of each pixel in the x and y dimensions
    size = max(x_max - x_min, y_max - y_min) / res_im
    
    # determine longer axis
    max_axis_x = (x_max - x_min) > (y_max - y_min)
    
    # iterate over each point in the point cloud and update the depth image
    if inverse:
        for point in points:
            
            # calculate the position of the point in the depth image
            if max_axis_x:
                x_pos = int((point[0] - x_min) / size)
                y_pos = int((point[1] - y_med) / size) + int(res_im/2)
            else:
                x_pos = int((point[0] - x_med) / size) + int(res_im/2)
                y_pos = int((point[1] - y_min) / size)
            
            # # check if index is out of bounds
            # if (x_pos > res_im - 1) or (y_pos > res_im -1):
            #     continue
            
            # update the corresponding pixel in the depth image with the z coordinate
            if (top_image[x_pos % res_im, y_pos % res_im] == -999) | (point[2] < top_image[x_pos % res_im, y_pos % res_im]): # wraped world to avoid out of range indexing
                top_image[x_pos % res_im, y_pos % res_im] = point[2]
    else:
        for point in points:
            
            # calculate the position of the point in the depth image
            if max_axis_x:
                x_pos = int((point[0] - x_min) / size)
                y_pos = int((point[1] - y_med) / size) + int(res_im/2)
            else:
                x_pos = int((point[0] - x_med) / size) + int(res_im/2)
                y_pos = int((point[1] - y_min) / size)
            
            # # check if index is out of bounds
            # if (x_pos > res_im - 1) or (y_pos > res_im -1):
            #     continue
            
            # update the corresponding pixel in the depth image with the z coordinate
            if point[2] > top_image[x_pos % res_im, y_pos % res_im]: # wraped world to avoid out of range indexing
                top_image[x_pos % res_im, y_pos % res_im] = point[2]
        
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
    x_min, y_min, z_min = np.min(points, axis = 0)
    x_max, y_max, z_max = np.max(points, axis = 0)
    x_med, y_med, z_med = np.median(points, axis = 0)
    
    # create an empty numpy array to store the depth image
    side_image = np.ones((res_im, res_im)) * -999
    
    # calculate the size of each pixel in the x and y dimensions
    size = max(x_max - x_min, z_max - z_min) / res_im
    
    # determine longer axis
    max_axis_x = (x_max - x_min) > (z_max - z_min)
    
    # iterate over each point in the point cloud and update the depth image
    for point in points:
        
        # calculate the position of the point in the depth image
        if max_axis_x:
            x_pos = int((point[0] - x_min) / size)
            z_pos = int((point[2] - z_med) / size) + int(res_im/2)
        else:
            x_pos = int((point[0] - x_med) / size) + int(res_im/2)
            z_pos = int((point[2] - z_min) / size)
        
        # # check if index is out of bounds
        # if x_pos > (res_im - 1) or z_pos > (res_im - 1):
        #     continue
        
        # update the corresponding pixel in the depth image with the z coordinate
        if point[1] > side_image[x_pos % res_im, z_pos % res_im]: # wraped world to avoid out of range indexing
            side_image[x_pos % res_im, z_pos % res_im] = point[1]
    
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
    if debug: print('n points in section section: ' + str(section.shape))
    
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
        section = section / np.max(abs(section))
        
        # create sideview
        section_image = sideview(section, res_im = res_im, plot = False)
        
    else:
        # create empty array
        section_image = np.zeros((res_im, res_im))
        
    # show image
    if plot:
        plt.imshow(section_image, interpolation = 'nearest')
        plt.show()
    
    # return array
    return section_image

#%%

# # execution for all training las files
# for path_curr in glob.glob(os.path.join(path_las, "*.las")):
#     points_to_images(path_curr)

# # example 
# las = lp.read(r"D:\Baumartenklassifizierung\data\train_downsampled\03498.las")
# points = np.stack((las.X, las.Y, las.Z), axis = 1)
# points = points * las.header.scale
# views = points_to_images(points, plot = True)
