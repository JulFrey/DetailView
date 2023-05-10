# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:43:42 2023

@author: Julian
"""

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
# class SimpleView(nn.Module):
#     def __init__(self, n_classes: int, n_views: int):
#         super().__init__()
        
#         # load backbone
#         backbone = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
        
#         # change first layer to greyscale
#         backbone.features[0].in_channels = 1
#         backbone.features[0].weight = torch.nn.Parameter(backbone.features[0].weight.sum(dim = 1, keepdim = True))
        
#         # remove effect of classifier
#         z_dim = backbone.classifier.in_features
#         backbone.classifier = nn.Identity()

#         # add new classifier
#         self.backbone = backbone
#         self.classifier = nn.Linear(
#             in_features = z_dim * n_views,
#             out_features = n_classes)

#     def forward(self, inputs: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
#         b, v, c, h, w = inputs.shape
#         inputs = inputs.reshape(b * v, c, h, w)
#         z = self.backbone(inputs)
#         z = z.reshape(b, v, -1)
#         z = z.reshape(b, -1)
#         return self.classifier(z)

# https://github.com/isaaccorley/simpleview-pytorch/blob/main/simpleview_pytorch/simpleview.py
class SimpleView(nn.Module):
    def __init__(self, n_classes: int, n_views: int):
        super().__init__()
        
        # load backbone
        backbone = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
        
        # change first layer to greyscale
        backbone.features[0].in_channels = 1
        backbone.features[0].weight = torch.nn.Parameter(backbone.features[0].weight.sum(dim = 1, keepdim = True))
        
        # remove effect of classifier
        z_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()

        # add new classifier & float pathway
        self.backbone = backbone
        self.float_pathway = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
            nn.ReLU())
        self.classifier = nn.Linear(
            in_features = z_dim * (n_views + 1),
            out_features = n_classes)

    def forward(self, inputs: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = inputs.shape
        inputs = inputs.reshape(b * v, c, h, w)
        img = self.backbone(inputs)
        img = img.reshape(b, v, -1).reshape(b, -1)
        hei = self.float_pathway(heights.view(-1, 1))
        img = torch.cat((img, hei), dim = 1)
        return self.classifier(img)
