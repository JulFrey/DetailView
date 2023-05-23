# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:43:42 2023

@author: Julian
"""

# import packages
import torch
import torchvision
import torch.nn as nn

class ParallelDenseNet(nn.Module):
    def __init__(self, n_classes: int, n_views: int):
        super(ParallelDenseNet, self).__init__()
        
        # define a single DenseNet
        self.shared_densenet = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
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
        self.classifier = nn.Sequential(
            nn.Linear(in_features = z_dim * (n_views + 1), out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = n_classes))

    def forward(self, inputs: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
        
        # pass each input tensor through the shared DenseNet
        img1 = self.shared_densenet(inputs[:,0,:,:,:])
        img2 = self.shared_densenet(inputs[:,1,:,:,:])
        img3 = self.shared_densenet(inputs[:,2,:,:,:])
        img4 = self.shared_densenet(inputs[:,3,:,:,:])
        img5 = self.shared_densenet(inputs[:,4,:,:,:])
        img6 = self.shared_densenet(inputs[:,5,:,:,:])
        img7 = self.shared_densenet(inputs[:,6,:,:,:])
        del inputs
        
        # concatenate output tensors from all branches
        img = torch.cat((img1, img2, img3, img4, img5, img6, img7), dim = 1)
        img = self.flatten(img)
        
        # using height float
        heights = self.float_pathway(heights.view(-1, 1))

        # concatenate float input to flattened tensor
        img = torch.cat((img, heights), dim = 1)

        # pass the concatenated tensor through fully connected layers
        label = self.classifier(img)
        
        # return predicted label
        return label

# # https://github.com/isaaccorley/simpleview-pytorch/blob/main/simpleview_pytorch/simpleview.py
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
