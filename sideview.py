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

# set paths & variables
path_las = r"D:\Baumartenklassifizierung\data\processed\03498.las"
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

# prepare empty image
img = np.zeros((res_im, res_im))

# ??? :(
