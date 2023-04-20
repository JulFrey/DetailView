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
import matplotlib.pyplot as plt

# set paths & variables
path_las = r"D:\Baumartenklassifizierung\data\processed\03498.las"
path_out = r"D:\Baumartenklassifizierung\data\images"

# create output folder
if not os.path.exists(path_out):
   os.makedirs(path_out)

#%%

def points_to_images(path_las, res_im = 256, num_side = 4, plot = False):
    """

    Parameters
    ----------
    path_las : string
        path to las file
    res_im : int, optional
        edge length of the quadratic tensor. The default is 256.
    num_side : int, optional
        number of side views. The default is 4.
    plot : bool, optional
        Plot the results for debugging. The default is False.

    Returns
    -------
    views : TYPE
        DESCRIPTION.

    """
    
    # read in las file
    las = lp.read(path_las)
    
    # turn coordinates into numpy array
    points = np.stack((las.X, las.Y, las.Z), axis = 1)
    points = points * las.header.scale
    
    # will be done by augmentation function >>
    
    # center point cloud
    points = points - np.median(points, axis = 0)
    
    # scale point cloud
    points = points / np.max(abs(points))
    
    # << will be done by augmentation function
    
    # prepare view array
    views = np.zeros((num_side + 2, res_im, res_im))
    
    # add top view
    views[0,:,:] = topview(points, res_im, plot = plot)
    
    # loop through perspectives
    deg_steps = np.linspace(0, 180, num = num_side)
    for i in range(num_side):
        
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
        views[i+1,:,:] = sideview(points_rot, res_im, plot = plot)
    
    # add bottom view
    views[num_side + 1,:,:] = topview(points, res_im, inverse = True, plot = plot)
    
    # return
    return views

# creating topview
def topview(points, res_im, inverse = False, plot = False):
    
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
def sideview(points, res_im, plot = False):
    
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

#%%

# # execution for all training las files
# for path_curr in glob.glob(os.path.join(path_las, "*.las")):
#     points_to_images(path_curr)

# example 
views = points_to_images(r"D:\TLS\Puliti_Reference_Dataset\train\03492.las", plot = True)
