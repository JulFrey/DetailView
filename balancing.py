# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:13:17 2023

@author: Zoe
"""

# import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, scale

#%% helper function

# vector distance
def arr_point_dist(array, point):
    # return np.sqrt(np.sum((array - point)**2, axis = 1))
    return np.linalg.norm(array - point, axis = 1)

#%% prepare labels

# set validation dataset length
validation_len = 400

# read the csv file with labels to convert
labels = pd.read_csv(r"S:\3D4EcoTec\tree_metadata_training_publish.csv")

#%% pick test data using farthest distance sampling

# TODO: does not work :(

# add height as height class
labels["height"] = np.floor(labels["tree_H"].to_numpy()/5)

# encode values as numbers
le = LabelEncoder(); labels["num_species"]   = scale(le.fit_transform(labels["species"]))
le = LabelEncoder(); labels["num_data_type"] = scale(le.fit_transform(labels["data_type"]))
le = LabelEncoder(); labels["num_height"]    = scale(le.fit_transform(labels["height"]))

# extract as numpy array
parameters = labels[["num_species", "num_data_type", "num_height"]].values

# get random first point
first_idx = np.random.randint(0, parameters.shape[0])

# calculate distances from first point to all other points
distances = arr_point_dist(parameters, parameters[first_idx,:])

# loop through all points
selected_indices = [first_idx]
for i in range(1, validation_len):
    
    # sample indices dependig on distance from selected point
    far_away_idx = np.random.choice(np.arange(len(distances)), p = (distances**2)/np.sum(distances**2))
    
    # add selected point index to list
    selected_indices.append(far_away_idx)
    
    # get distances to new point
    new_distances = arr_point_dist(parameters, parameters[far_away_idx,:])
    
    # update distances with minimum distance between old and new distances
    distances = np.minimum(distances, new_distances)
    print(np.max(distances))

# show results
selected_indices
selected_indices.shape

#%% R CODE

# # combine data in one data frame
# parameter_mat <- with(labels, data.frame(X = scale(as.integer(as.factor(species))), Y =  scale(as.integer(as.factor(data_type))), Z = scale(tree_H)))

# # weighing by distances
# farthest_point_sampling_idx_rand <- function(points, k) {
#   # Initialize list to store selected point indices
#   selected_indices <- sample(1:nrow(points), 1)
  
#   # Calculate distances from first point to all other points
#   distances <- apply(points, 1, function(x) norm_vec(x - points[selected_indices[1], ]))
  
#   # Select remaining k-1 points
#   for (i in 2:k) {

#     # Find index of point with maximum distance from selected points
#     # max_dist_index <- which.max(distances)
#     max_dist_index <- sample(1:length(distances), 1, prob = distances^2)
    
#     # Add selected point index to list
#     selected_indices <- c(selected_indices, max_dist_index)
    
#     # Update distances with new selected point
#     new_distances <- apply(points, 1, function(x) norm_vec(x - points[max_dist_index, ]))
    
#     # Update distances with minimum distance between old and new distances
#     distances <- pmin(distances , new_distances)
#     #dist_vec <<- distances
#   }
  
#   return(selected_indices)
# }

# # normalize vector
# norm_vec <- function(x){
#   sqrt(crossprod(x))[1]
# }

#%% derive weights for training data

# TODO: get train data points only

# get class counts
count_species = labels["species"].value_counts().to_dict()
count_sensor  = labels["data_type"].value_counts().to_dict()
count_height  = labels["height"].value_counts().to_dict()

# add count data as a column
labels["n_species"]   = labels["species"].map(count_species)
labels["n_data_type"] = labels["data_type"].map(count_sensor)
labels["n_height"]    = labels["height"].map(count_height)

# derive weights
labels["w_species"]   = 1 - labels["n_species"] / labels.shape[0]
labels["w_data_type"] = 1 - labels["n_data_type"] / labels.shape[0]
labels["w_height"]    = 1 - labels["n_height"] / labels.shape[0]

# produce total weight
labels["weight"] = labels["w_species"] * labels["w_data_type"] * labels["w_height"]
# TODO: maybe we have to rescale this?

# # save new label data frame
# labels.to_csv(r"S:\3D4EcoTec\train_labels.csv", index = False)