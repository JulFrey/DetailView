# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:26:11 2023

@author: Julian
"""

# import packages
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from skimage import io, transform

# import own functions
import augmentation as au
import sideview as sv

# # get model input for one point cloud
# pts = au.augment(r"D:\Baumartenklassifizierung\data\train_downsampled\03498.las")
# views = sv.points_to_images(pts)
# tensor = torch.tensor(views)

# read the csv file with labels to convert
labels = pd.read_csv(r"S:\3D4EcoTec\tree_metadata_training_publish.csv")

# initialize LabelEncoder object
le = LabelEncoder()

# transform the string labels to integer labels
labels = pd.concat([labels, pd.DataFrame(le.fit_transform(labels['species']), columns=["species_id"])], axis = 1)
labels = labels[['filename', 'species_id', 'tree_H']]

# TODO: change path to downsampled point clouds?

# save new label data frame
labels.to_csv(r"S:\3D4EcoTec\train_labels.csv", index = False)

#%%

# create dataset class to load the data from csv and las files
class TrainDataset():
    
    """Tree species dataset."""
    
    # initialization
    def __init__(self, csv_file, root_dir, img_trans = None): # TODO: change variable name to not overwrite loaded function?
        
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations with the collumns
                0: filenme, 1: label_id, 2: tree height.
            root_dir (string): Directory with all the las files.
            img_trans (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # set attributes
        self.trees_frame = pd.read_csv(csv_file)
        self.root_dir    = root_dir
        self.img_trans   = img_trans
    
    # length
    def __len__(self):
        return len(self.trees_frame)
    
    # indexing
    def __getitem__(self, idx):
        
        # convert indices to list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get full las path
        las_name = os.path.join(
            self.root_dir,
            *self.trees_frame.iloc[idx, 0].split('/'))
        views = sv.points_to_images(au.augment(las_name))
        
        # get side views
        views = torch.from_numpy(views)
        
        # augment images
        if self.img_trans:
            views = self.img_trans(views)
        
        # return images with labels
        return {'views': views,
                'species': self.trees_frame.iloc[idx, 1],
                'height': self.trees_frame.iloc[idx, 2]}

# %%

# create dataset object
tree_dataset = TrainDataset(r"S:\3D4EcoTec\train_labels.csv", r"S:\3D4EcoTec\downsampled")

# visualizing  sample
test = tree_dataset[1]
plt.imshow(test["views"][0,:,:], interpolation = 'nearest')
plt.show()

#%%

# setting up image augmentation
data_transform = transforms.Compose([
    transforms.RandomCrop(200)
    ])

# create dataset object
tree_dataset = TrainDataset(r"S:\3D4EcoTec\train_labels.csv", r"S:\3D4EcoTec\downsampled", img_trans = data_transform) 

# visualizing  sample
test = tree_dataset[1]
plt.imshow(test["views"][0,:,:], interpolation = 'nearest')
plt.show()

# %%

class TrainDataset_testing():
    """Tree species dataset."""

    # initialization
    def __init__(self, csv_file, root_dir, img_trans = None): # TODO: change variable name to not overwrite loaded function?
        
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations with the collumns
                0: filenme, 1: label_id, 2: tree height.
            root_dir (string): Directory with all the las files.
            img_trans (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # set attributes
        self.trees_frame = pd.read_csv(csv_file)
        self.root_dir    = root_dir
        self.img_trans   = img_trans
    
    # length
    def __len__(self):
        return len(self.trees_frame)
    
    # indexing
    def __getitem__(self, idx):
        
        # convert indices to list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get full las path
        las_name = os.path.join(
            self.root_dir,
            *self.trees_frame.iloc[idx, 0].split('/'))
        views = sv.points_to_images(au.augment(las_name))
        
        # get side view
        views = torch.from_numpy(views[1,:,:])
        
        # augment images
        if self.img_trans:
            views = self.img_trans(views)
        
        # return images with labels
        return {'views': views,
                'species': self.trees_frame.iloc[idx, 1]}

def collate_fn(list_items):
     x = []
     y = []
     for X in list_items:
     #     #print(f'x_={views_}, y_={y_}')
          x.append(X["views"])
          y.append(X["species"])
     return x, y

#%%

# get the cuda device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
print(f"Using {device} device")

# load empty densenet201 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights = None)

# ???
model.to('cuda')

# change input and output layer
model.classifier = torch.nn.Linear(1024, 33)
model.features[0] = torch.nn.Conv2d(1, 64, kernel_size = (7,7))

# create dataset object
tree_dataset = TrainDataset_testing(r"S:\3D4EcoTec\train_labels.csv", r"S:\3D4EcoTec\downsampled")

# create data loader
batch_size = 2
data_loader = torch.utils.data.DataLoader(tree_dataset, batch_size, collate_fn = collate_fn)

for tensor, label in data_loader:  
    sample_image = tensor   # Reshape them according to your needs.
    sample_label = label
    break
