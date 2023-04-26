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
import torchvision
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
#from skimage import io, transform

# import own functions
import augmentation as au
import sideview as sv

#%% prepare labels

# read the csv file with labels to convert
labels = pd.read_csv(r"S:\3D4EcoTec\tree_metadata_training_publish.csv")

# initialize LabelEncoder object
le = LabelEncoder()

# transform the string labels to integer labels
labels = pd.concat([labels, pd.DataFrame(le.fit_transform(labels['species']), columns = ["species_id"])], axis = 1)
labels = labels[['filename', 'species_id', 'tree_H']]

# check if files exist
exists = []
for p in r'S:\3D4EcoTec\down' + labels['filename']:
    exists.append(os.path.exists(p))

# exclude rows refering to not existing files
labels = labels[pd.Series(exists)]

# save new label data frame
labels.to_csv(r"S:\3D4EcoTec\train_labels.csv", index = False)

# TODO: split in training & validation data

#%% prepare dataset class

# create dataset class to load the data from csv and las files
class TrainDataset_AllChannels():
    
    """Tree species dataset."""
    
    # initialization
    def __init__(self, csv_file, root_dir, img_trans = None):
        
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
        image = sv.points_to_images(au.augment(las_name))
        
        # get side views
        image = torch.from_numpy(image)
        
        # augment images
        if self.img_trans:
            image = self.img_trans(image)
        
        # get height
        height = torch.tensor(self.trees_frame.iloc[idx, 2], dtype = torch.float)
        
        # get species
        label = torch.tensor(self.trees_frame.iloc[idx, 1] + 1, dtype = torch.int64)
        
        # return images with labels
        return image, height, label

#%% test dataset class without augmentation

# create dataset object
dataset = TrainDataset_AllChannels(r"S:\3D4EcoTec\train_labels.csv", r"S:\3D4EcoTec")

# visualizing  sample
test = dataset[1]
plt.imshow(test[0][0,:,:], interpolation = 'nearest')
plt.show()

#%% test dataset class with augmentation

# setting up image augmentation
trafo = transforms.Compose([
    transforms.RandomCrop(200),
    ])

# create dataset object
dataset = TrainDataset_AllChannels(r"S:\3D4EcoTec\train_labels.csv", r"S:\3D4EcoTec\down", img_trans = trafo) 

# visualizing  sample
test = dataset[1]
plt.imshow(test[0][0,:,:], interpolation = 'nearest')
plt.show()

#%% set up new dataset class for testing

class TrainDataset_SingleChannel():
    """Tree species dataset."""

    # initialization
    def __init__(self, csv_file, root_dir, img_trans = None, channel = 0):
        
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
        self.channel     = channel
    
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
        image = sv.points_to_images(au.augment(las_name))
        
        # get selected side & top views
        image = torch.from_numpy(image[self.channel,:,:])
        image = torch.unsqueeze(image, dim = 0)
        
        # augment images
        if self.img_trans:
            image = self.img_trans(image)
        
        # get height
        height = torch.tensor(self.trees_frame.iloc[idx, 2], dtype = torch.float)
        
        # get species
        label = torch.tensor(self.trees_frame.iloc[idx, 1] + 1, dtype = torch.int64)
        
        # return images with labels
        return image, height, label

#%% test dataloader without augmentation

# create dataset object
dataset = TrainDataset_SingleChannel(r"S:\3D4EcoTec\train_labels.csv", r"S:\3D4EcoTec\down")

# create data loader
batch_size = 16
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

# # test output of iterator
# image, height, label = next(iter(dataloader))
# print(image.shape); print(height.shape); print(label.shape)

# # show image
# plt.imshow(image[0,0,:,:], interpolation = 'nearest')
# plt.show()

#%% test dataloader with augmentation

# setting up image augmentation
trafo = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(64), # for testing the model
    transforms.ToTensor()
])

# create dataset object
dataset = TrainDataset_SingleChannel(r"S:\3D4EcoTec\train_labels.csv", r"S:\3D4EcoTec\down", img_trans = trafo)

# create data loader
batch_size = 16
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True) # num_workers = 5, pin_memory = True

# # test output of iterator
# image, height, label = next(iter(dataloader))
# print(image.shape); print(height.shape); print(label.shape)

# # show image
# plt.imshow(image[0,0,:,:], interpolation = 'nearest')
# plt.show()

#%% training cnn

# load the model
model = torchvision.models.densenet201() # weights = 'DenseNet201_Weights.DEFAULT')

# change first & last layer
model.features[0] = torch.nn.Conv2d(1, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False) # change number of input channels
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, int(len(le.classes_) + 1))

# get the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

# give to devise
model.to(device)

# define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# loop through epochs
num_epochs = 2
for epoch in range(num_epochs):
    
    #  model.train() # ich check pytorch nicht, aber sollte das nicht irgendwo auftauchen?
    running_loss = 0.0
    
    # loop through whole dataset?
    for i, data in enumerate(dataloader, 0): 
        
        # print progress every ten batches
        if i % 10 == 9:
            progress = (i / (len(dataset) / batch_size))
            print('[epoch: %d] dataset: %.2f%%' %
                  (epoch + 1, progress * 100))
        # load data
        inputs, height, labels = data
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics every 100 batches
        running_loss += loss.item()
        if i % 100 == 99:
            print('[epoch: %d, batch: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished training')
