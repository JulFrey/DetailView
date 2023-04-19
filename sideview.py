# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
import os
import math
import glob
import laspy as lp
import numpy as np
import matplotlib.pyplot as plt
import torch

# set paths & variables
path_las = r"D:\TLS\Puliti_Reference_Dataset\train\03498.las"
path_out = r"D:\Baumartenklassifizierung\data\raw"

# create output folder
if not os.path.exists(path_out):
   os.makedirs(path_out)

#%%

# TODO: wrap in a function

# set parameters
num_perspective = 4
res_im = 250

# read in las file
las = lp.read(path_las)

# turn coordinates into numpy array
points = np.stack((las.X, las.Y, las.Z), axis = 1)
points = points * las.header.scale

# # loop through perspectives
# for deg in np.linspace(0, 180, num = num_perspective):
#     print(deg)
deg = 60

# z rotation matrix
r_z_rad = math.radians(deg)
r_z_mat = np.array([
    [math.cos(r_z_rad), -math.sin(r_z_rad), 0],
    [math.sin(r_z_rad),  math.cos(r_z_rad), 0],
    [                0,                  0, 1]])

# rotate point cloud
points_rotated = np.matmul(points, r_z_mat.T)

# center point cloud
# (should happen in augmentation process)
points = points - np.median(points, axis = 0)

# scale point cloud
points = points / np.max(abs(points))

#%%

# momentane lösung staucht die Achsen zuammen :(

#%%

# creating topview

# create an empty numpy array to store the depth image
depth_image = np.ones((res_im, res_im)) * -999

# find the minimum and maximum values of the x, y, and z coordinates
x_min, y_min, z_min = np.min(points, axis = 0)
x_max, y_max, z_max = np.max(points, axis = 0)

# calculate the size of each pixel in the x and y dimensions
size = (max(x_max, y_max) - min(x_min,y_min)) / res_im
#y_size = (max(x_max, y_max) - min(x_min,y_min)) / res_im

# iterate over each point in the point cloud and update the depth image
for point in points:
    
    # calculate the position of the point in the depth image
    x_pos = int((point[0] - x_min) / size) - 1
    y_pos = int((point[1] - y_min) / size) - 1
    
    # update the corresponding pixel in the depth image with the z coordinate
    if point[2] > depth_image[x_pos, y_pos]:
        depth_image[x_pos, y_pos] = point[2]

# replace dummy values with value
depth_image[depth_image == -999] = 0

# show image
plt.imshow(depth_image, interpolation='nearest')
plt.show()

#%%

# creating sideview

# create an empty numpy array to store the depth image
depth_image = np.ones((res_im, res_im)) * -999

# find the minimum and maximum values of the x, y, and z coordinates
x_min, y_min, z_min = np.min(points, axis = 0)
x_max, y_max, z_max = np.max(points, axis = 0)

# calculate the size of each pixel in the x and y dimensions
size = (max(x_max, z_max) - min(x_min,z_min)) / res_im
#y_size = (max(x_max, y_max) - min(x_min,y_min)) / res_im

# iterate over each point in the point cloud and update the depth image
for point in points:
    
    # calculate the position of the point in the depth image
    x_pos = int((point[0] - x_min) / size) - 1
    z_pos = int((point[2] - z_min) / size) - 1
    
    # update the corresponding pixel in the depth image with the z coordinate
    if point[1] > depth_image[x_pos, z_pos]:
        depth_image[x_pos, z_pos] = point[1]

# replace dummy values with value
depth_image[depth_image == -999] = 0

# show image
plt.imshow(depth_image, interpolation='nearest')
plt.show()