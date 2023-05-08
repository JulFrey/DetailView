# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:43:42 2023

@author: Julian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(ParallelDenseNet, self).__init__()
        self.num_classes = num_classes
        
        # Define a single DenseNet
        self.shared_densenet = torch.hub.load('pytorch/vision', 'densenet201', weights = 'DenseNet201_Weights.DEFAULT')
        self.shared_densenet.features[0].in_channels = 1
        self.shared_densenet.features[0].weight = nn.Parameter(self.shared_densenet.features[0].weight.sum(dim = 1, keepdim = True))
        self.shared_densenet = nn.Sequential(*list(self.shared_densenet.features.children()))
        
        # Define fully connected layers
        self.fc1 = nn.Linear(430080, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes)
        self.flatten = nn.Flatten()

    def forward(self, inputs, heights):
        
        # Pass each input tensor through the shared DenseNet
        img1 = self.shared_densenet(inputs[:,0,:,:,:])
        img2 = self.shared_densenet(inputs[:,1,:,:,:])
        img3 = self.shared_densenet(inputs[:,2,:,:,:])
        img4 = self.shared_densenet(inputs[:,3,:,:,:])
        img5 = self.shared_densenet(inputs[:,4,:,:,:])
        img6 = self.shared_densenet(inputs[:,5,:,:,:])
        img7 = self.shared_densenet(inputs[:,6,:,:,:])
        del inputs
        
        # Concatenate the output tensors from all branches
        img = torch.cat((img1, img2, img3, img4, img5, img6, img7), dim = 1)
        img = self.flatten(img)

        # Broadcast the float input tensor to match the size of x
        heights = heights.view(-1, 1).expand(-1, img.shape[1])

        # Concatenate the float input to the flattened tensor
        img = torch.cat((img, heights), dim = 1)

        # Pass the concatenated tensor through fully connected layers
        img = self.fc1(img)
        img = F.relu(img)
        img = self.fc2(img)

        return img
