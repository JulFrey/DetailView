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
        self.shared_densenet = torch.hub.load('pytorch/vision', 'densenet201', pretrained=False)
        self.shared_densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.shared_densenet = nn.Sequential(*list(self.shared_densenet.features.children()))
        # Define fully connected layers
        self.fc1 = nn.Linear(1474559, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x1, x2, x3, x4, x5, x6, y):
        # Pass each input tensor through the shared DenseNet
        x1 = self.shared_densenet(x1)
        x2 = self.shared_densenet(x2)
        x3 = self.shared_densenet(x3)
        x4 = self.shared_densenet(x4)
        x5 = self.shared_densenet(x5)
        x6 = self.shared_densenet(x6)

        # Concatenate the output tensors from all branches
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.flatten(x)

        # Broadcast the float input tensor to match the size of x
        y = y.view(-1, 1)
        y = y.expand(-1, x.shape[1] - 1)

        # Concatenate the float input to the flattened tensor
        x = torch.cat((x, y), dim=1)

        # Pass the concatenated tensor through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Define the model and loss function
model = ParallelDenseNet(num_classes=33)
criterion = nn.SmoothL1Loss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)