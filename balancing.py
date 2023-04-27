# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:13:17 2023

@author: Zoe
"""

# import packages
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, scale

#%% prepare labels

# set validation dataset length
validation_len = 400

# read the csv file with labels to convert
labels = pd.read_csv(r"S:\3D4EcoTec\tree_metadata_training_publish.csv")

#%% remove not existing files

# encode species & data type as numbers
le = LabelEncoder(); labels["species_id"] = le.fit_transform(labels["species"])
le = LabelEncoder(); labels["data_type_id"] = le.fit_transform(labels["data_type"])

# check if files exist
exists = [os.path.exists("S:/3D4EcoTec/down" + filename) for filename in labels["filename"]]

# exclude rows refering to not existing files
labels = labels[exists]

#%% pick test data using farthest distance sampling

# vector distance
def arr_point_dist(array, point):
    # return np.sqrt(np.sum((array - point)**2, axis = 1))
    return np.linalg.norm(array - point, axis = 1)

# scale values
labels["s_species"]   = scale(labels["species_id"])
labels["s_data_type"] = scale(labels["data_type_id"])
labels["s_height"]    = scale(labels["tree_H"])

# extract as numpy array
parameters = labels[["s_species", "s_data_type", "s_height"]].values

# get random first point
selected_indices = [np.random.randint(0, parameters.shape[0])]

# calculate distances from first point to all other points
distances = arr_point_dist(parameters, parameters[selected_indices[0],:])

# loop through all points
for i in range(1, validation_len):
    
    # sample indices dependig on distance from selected point
    far_away_idx = np.random.choice(np.arange(len(distances)), p = (distances**2)/np.sum(distances**2))
    
    # add selected point index to list
    selected_indices.append(far_away_idx)
    
    # get distances to new point
    new_distances = arr_point_dist(parameters, parameters[far_away_idx,:])
    
    # update distances with minimum distance between old and new distances
    distances = np.minimum(distances, new_distances)

# subset data based on indices
vali = labels[labels.index.isin(selected_indices)].copy()
train = labels[~labels.index.isin(selected_indices)].copy()

# save test data
vali = vali[["filename", "species_id", "tree_H"]]
vali.to_csv(r"S:\3D4EcoTec\vali_labels.csv", index = False)

#%% derive weights for training data

# add height as height class
train["height"] = np.floor(train["tree_H"].to_numpy()/5)

# get class counts
count_species = train["species"].value_counts().to_dict()
count_sensor  = train["data_type"].value_counts().to_dict()
count_height  = train["height"].value_counts().to_dict()

# add count data as a column
train["n_species"]   = train["species"].map(count_species)
train["n_data_type"] = train["data_type"].map(count_sensor)
train["n_height"]    = train["height"].map(count_height)

# derive weights
train["w_species"]   = 1 - train["n_species"] / train.shape[0]
train["w_data_type"] = 1 - train["n_data_type"] / train.shape[0]
train["w_height"]    = 1 - train["n_height"] / train.shape[0]

# produce total weight
train["weight"] = train["w_species"] * train["w_data_type"] * train["w_height"]
train["weight"] = train["weight"] / sum(train["weight"])

# save new label data frame
train = train[["filename", "species_id", "tree_H", "weight"]]
train.to_csv(r"S:\3D4EcoTec\train_labels.csv", index = False)
