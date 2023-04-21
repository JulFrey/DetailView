# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:26:11 2023

@author: Julian
"""

import torch
import augmentation as au
import sideview as sw
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from skimage import io, transform

pts = au.augment(r"D:\TLS\Puliti_Reference_Dataset\train\03492.las")
#views = sw.points_to_images(pts)
#tensor = torch.tensor(views)

# read the csv file with labels to convert
labels = pd.read_csv(r"D:\TLS\Puliti_Reference_Dataset\tree_metadata_training_publish.csv")

# Initialize the LabelEncoder object
le = LabelEncoder()

# Fit and transform the string labels to integer labels
labels = pd.concat([labels, pd.DataFrame(le.fit_transform(labels['species']), columns=["species_id"])], axis = 1)
labels = labels[['filename', 'species_id','tree_H']]
labels.to_csv(r"D:\TLS\Puliti_Reference_Dataset\train_labels.csv", index=False)


#%%
# create dataset class to load the data from csv and las files
class TrainDataset():
    """Tree species dataset."""

    def __init__(self, csv_file, root_dir, transform = None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations with the collumns
                0: filenme, 1: label_id, 2: tree height.
            root_dir (string): Directory with all the las files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.trees_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.trees_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        las_name = os.path.join(self.root_dir,
                                *self.trees_frame.iloc[idx, 0].split('/'))
        views = sw.points_to_images(au.augment(las_name))
        views = torch.from_numpy(views)
        
        if self.transform:
            views = self.transform(views)
        
        sample = {'views': views, 'species': self.trees_frame.iloc[idx, 1], 'height': self.trees_frame.iloc[idx, 2]}

        return sample

tree_dataset = TrainDataset(r"D:\TLS\Puliti_Reference_Dataset\train_labels.csv", "D:\\\\TLS\\Puliti_Reference_Dataset") 

# test = tree_dataset[1]
# plt.imshow(test["views"][0,:,:], interpolation = 'nearest')
# plt.show()

# data_transform = transforms.Compose([
#         transforms.RandomCrop(200)
#     ])

# tree_dataset = TrainDataset(r"D:\TLS\Puliti_Reference_Dataset\train_labels.csv", "D:\\\\TLS\\Puliti_Reference_Dataset", data_transform) 

# test = tree_dataset[1]
# plt.imshow(test["views"][0,:,:])
# plt.show()

# get the cuda device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights=None)
model.to('cuda')

# change input and output layer
model.classifier = torch.nn.Linear(1024, 33)
model.features[0] = torch.nn.Conv2d(1, 64, kernel_size = (7,7))

class TrainDataset_testing():
    """Tree species dataset."""

    def __init__(self, csv_file, root_dir, transform = None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations with the collumns
                0: filenme, 1: label_id, 2: tree height.
            root_dir (string): Directory with all the las files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.trees_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.trees_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        las_name = os.path.join(self.root_dir,
                                *self.trees_frame.iloc[idx, 0].split('/'))
        views = sw.points_to_images(au.augment(las_name))
        views = torch.from_numpy(views[1,:,:])
        
        if self.transform:
            views = self.transform(views)
        
        sample = {'views': views, 'species': self.trees_frame.iloc[idx, 1]}

        return sample

tree_dataset = TrainDataset_testing(r"D:\TLS\Puliti_Reference_Dataset\train_labels.csv", "D:\\\\TLS\\Puliti_Reference_Dataset")
batch_size = 2

def collate_fn(list_items):
     x = []
     y = []
     for X in list_items:
     #     #print(f'x_={views_}, y_={y_}')
          x.append(X["views"])
          y.append(X["species"])
     return x, y


data_loader = torch.utils.data.DataLoader(tree_dataset, batch_size, collate_fn= collate_fn)

for tensor, label in data_loader:  
    sample_image = tensor   # Reshape them according to your needs.
    sample_label = label
    break
#%%


# Define the transform to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Define the dataset and data loader
dataset = TrainDataset_testing(r"D:\TLS\Puliti_Reference_Dataset\train_labels.csv", "D:\\\\TLS\\Puliti_Reference_Dataset", transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn = collate_fn)

# Define the model
model = torchvision.models.densenet201(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, 33)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished training')
