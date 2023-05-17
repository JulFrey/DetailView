# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Zoe
"""

# import packages
import os
import glob
import pdal
import pandas as pd

# # set paths & variables
# path_las = r"S:\3D4EcoTec\train"
# path_out = r"C:\TLS\down\train"

# set paths & variables
path_las = r"S:\3D4EcoTec\test"
path_out = r"C:\TLS\down\test"

# create output folder
if not os.path.exists(path_out):
   os.makedirs(path_out)

#%%

def downsample(path_las, path_out, res_pc = 0.01, min_n = 100, get_height = False):
    
    """
    Parameters
    ----------
    path_las : str
        Path to the input las file.
    path_out : str
        Path to the output folder.
    res_pc : float, optional
        Target las resolution. The default is 0.01.
    min_n : float, optional
        Minimum number of points. The default is 100.
    
    Returns
    -------
    Path to new point cloud.
    """

    # get output path
    path_out_full = os.path.join(path_out, os.path.basename(path_las))
    print(path_out_full)
    
    # setting up downsample
    las_reader = pdal.Reader(path_las, type = "readers.las", nosrs = True)
    las_filter = pdal.Filter(type = "filters.voxelcentroidnearestneighbor", cell = res_pc)
    las_writer = pdal.Writer(path_out_full, dataformat_id = 0)
    
    # get height
    if get_height:
        pipeline = pdal.Pipeline([las_reader])
        pipeline.execute()
        height = pipeline.arrays[0]["Z"].max() - pipeline.arrays[0]["Z"].min()
    else:
        height = 0
    
    # get number of points
    pipeline = pdal.Pipeline([las_reader])
    pipeline.execute()
    n = pipeline.metadata["metadata"]['readers.las']['count']
    
    # skip if too few points
    if n < min_n:
        return ""
    
    # downsample point cloud
    pipeline = las_reader | las_filter | las_writer
    pipeline.execute()
        
    # return path
    return path_out_full, height

#%% train data

# # execution for all training las files
# for path_curr in glob.glob(os.path.join(path_las, "*.las")):
#     downsample(path_curr, path_out)

#%% test data

# create dataframe
df = pd.DataFrame({"filename": [], "species_id": [], "tree_H": []})

# execution for all training las files
for path_curr in glob.glob(os.path.join(path_las, "*.las")):
    
    # downsample & get height
    path_sub, height = downsample(path_curr, path_out, min_n = 1, get_height = True)

    # change path
    lowest_folder = os.path.basename(os.path.dirname(path_sub))
    filename = os.path.basename(path_sub)
    path_sub = os.path.join(lowest_folder, filename)
    
    # append dataframe
    curr = pd.DataFrame({"filename": [path_sub], "species_id": [-999], "tree_H": [height]})
    df = pd.concat([df, curr], ignore_index = True)
        
# save as csv
csv_path = os.path.join(os.path.dirname(path_out), "test_labels.csv")
df.to_csv(csv_path, index = False)
