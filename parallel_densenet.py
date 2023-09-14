# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:43:42 2023

@author: Julian
"""

# import packages
import torch
import torchvision
import torch.nn as nn
import pandas as pd
import os
import numpy as np

# import own scripts
import augmentation as au
import sideview as sv
import read_las as rl

# https://github.com/isaaccorley/simpleview-pytorch/blob/main/simpleview_pytorch/simpleview.py
class SimpleView(nn.Module):
    def __init__(self, n_classes: int, n_views: int):
        super().__init__()
        
        # load model for sides views
        sides = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
        
        # change first layer to greyscale
        sides.features[0].in_channels = 1
        sides.features[0].weight = torch.nn.Parameter(sides.features[0].weight.sum(dim = 1, keepdim = True))
        
        # remove effect of classifier
        z_dim = sides.classifier.in_features
        sides.classifier = nn.Identity()
        
        # load model for tops views
        tops = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
        
        # change first layer to greyscale
        tops.features[0].in_channels = 1
        tops.features[0].weight = torch.nn.Parameter(tops.features[0].weight.sum(dim = 1, keepdim = True))
        
        # remove effect of classifier
        tops.classifier = nn.Identity()
        
        # load model for details
        details = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
        
        # change first layer to greyscale
        details.features[0].in_channels = 1
        details.features[0].weight = torch.nn.Parameter(details.features[0].weight.sum(dim = 1, keepdim = True))
        
        # remove effect of classifier
        details.classifier = nn.Identity()
        
        # add new classifier & float pathway
        self.sides_pathway = sides
        self.tops_pathway = tops
        self.details_pathway = details
        self.height_pathway = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
            nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(in_features = z_dim * (n_views + 1), out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = n_classes))

    def forward(self, inputs: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
        
        # prepare data
        b, v, c, h, w = inputs.shape
        sides = inputs[:,1:-2,:,:,:].reshape(b * (v - 3), c, h, w)
        tops = inputs[:,[0,-2],:,:,:].reshape(b * 2, c, h, w)
        details = inputs[:,-1,:,:,:].reshape(b * 1, c, h, w)
        del inputs
        
        # process sides views
        sides = self.sides_pathway(sides)
        sides = sides.reshape(b, (v - 3), -1).reshape(b, -1)
        
        # process tops views
        tops = self.tops_pathway(tops)
        tops = tops.reshape(b, 2, -1).reshape(b, -1)
        
        # process details
        details = self.details_pathway(details)
        details = details.reshape(b, 1, -1).reshape(b, -1)
        
        # process height
        heights = self.height_pathway(heights.view(-1, 1))
        
        # get label
        label = self.classifier(torch.cat((sides, tops, details, heights), dim = 1))
        return label

#%% create dataset class to load the data from csv and las files
class TrainDataset_AllChannels():
    
    """Tree species dataset."""
    
    # initialization
    def __init__(self, csv_file, root_dir, img_trans = None, pc_rotate = True,
                 height_noise = 0.01, height_mean = None,
                 height_sd = None, test = False, res = 512, n_sides = 4):
        
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations with the collumns
                0: filenme, 1: label_id, 2: tree height.
            root_dir (string): Directory with all the las files.
            img_trans (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # set attributes
        self.trees_frame  = pd.read_csv(csv_file)
        self.root_dir     = root_dir
        self.img_trans    = img_trans
        self.pc_rotate    = pc_rotate
        self.height_noise = height_noise
        self.height_mean  = height_mean
        self.height_sd    = height_sd
        self.test         = test
        self.res          = res
        self.n_sides      = n_sides
    
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
        
        # get side views
        if self.pc_rotate:
            image = sv.points_to_images(au.augment(las_name), res_im = self.res, num_side = self.n_sides)
        else:
            image = sv.points_to_images(rl.read_las(las_name), res_im = self.res, num_side = self.n_sides)
        image = torch.from_numpy(image)
        
        # augment images (all channels at once)
        if self.img_trans:
            image = self.img_trans(image)
        
        # add dimension
        image = image.unsqueeze(1)
        
        # get height
        height = torch.tensor(self.trees_frame.iloc[idx, 2], dtype = torch.float32)
        
        # augment height
        if self.height_noise > 0:
            height += np.random.normal(0, self.height_noise)
        
        # scale height using training mean & sd
        height = (height - self.height_mean) / self.height_sd
        
        # return images with filenames
        if self.test:
            las_path = self.trees_frame.iloc[idx, 0]
            return image, height, las_path
        
        # return images with labels
        label = torch.tensor(self.trees_frame.iloc[idx, 1], dtype = torch.int64)
        return image, height, label
    
    # training weights
    def weights(self):
        return torch.tensor(self.trees_frame["weight"].values)


