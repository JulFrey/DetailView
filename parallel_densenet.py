# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:43:42 2023

@author: Julian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelDenseNet(nn.Module):
    def __init__(self, num_classes, num_views):
        super(ParallelDenseNet, self).__init__()
        self.num_classes = num_classes
        self.num_views = num_views
        
        # define a single DenseNet
        self.shared_densenet = torch.hub.load('pytorch/vision', 'densenet201', weights = 'DenseNet201_Weights.DEFAULT')
        self.shared_densenet.features[0].in_channels = 1
        self.shared_densenet.features[0].weight = nn.Parameter(self.shared_densenet.features[0].weight.sum(dim = 1, keepdim = True))
        
        # remove effect of classifier
        z_dim = self.shared_densenet.classifier.in_features
        self.shared_densenet.classifier = nn.Identity()
        
        # define flattener
        self.flatten = nn.Flatten()
        
        # define float pathway
        self.float_pathway = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim))
        
        # create new classifier
        self.classifier = nn.Linear(
                   in_features = z_dim * (num_views + 1),
                   out_features = num_classes)

    def forward(self, inputs, heights):
        
        # pass each input tensor through the shared DenseNet
        img1 = self.shared_densenet(inputs[:,0,:,:,:])
        img2 = self.shared_densenet(inputs[:,1,:,:,:])
        img3 = self.shared_densenet(inputs[:,2,:,:,:])
        img4 = self.shared_densenet(inputs[:,3,:,:,:])
        img5 = self.shared_densenet(inputs[:,4,:,:,:])
        img6 = self.shared_densenet(inputs[:,5,:,:,:])
        img7 = self.shared_densenet(inputs[:,6,:,:,:])
        del inputs
        
        # concatenate the output tensors from all branches
        img = torch.cat((img1, img2, img3, img4, img5, img6, img7), dim = 1)
        img = self.flatten(img)
        
        # using height float
        heights = self.float_pathway(heights.view(-1, 1))

        # cncatenate the float input to the flattened tensor
        img = torch.cat((img, heights), dim = 1)

        # pass the concatenated tensor through fully connected layers
        label = self.classifier(img)

        return label
