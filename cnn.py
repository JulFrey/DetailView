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

# # get model input for one point cloud
# pts = au.augment(r"D:\Baumartenklassifizierung\data\train_downsampled\03498.las")
# views = sv.points_to_images(pts)
# tensor = torch.tensor(views)

# read the csv file with labels to convert
labels = pd.read_csv(r"V:\3D4EcoTec\tree_metadata_training_publish.csv")

# check if files exist
exists = []
for p in r'V:\3D4EcoTec\down' + labels['filename']:
    exists.append(os.path.exists(p))

labels = labels[pd.Series(exists)]

# initialize LabelEncoder object
le = LabelEncoder()

# transform the string labels to integer labels
labels = pd.concat([labels, pd.DataFrame(le.fit_transform(labels['species']), columns=["species_id"])], axis = 1)
labels = labels[['filename', 'species_id', 'tree_H']]

# TODO: change path to downsampled point clouds?
# TODO: exclude super small point clouds

# save new label data frame
labels.to_csv(r"V:\3D4EcoTec\train_labels.csv", index = False)

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
tree_dataset = TrainDataset(r"V:\3D4EcoTec\train_labels.csv", r"V:\3D4EcoTec")

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
tree_dataset = TrainDataset(r"V:\3D4EcoTec\train_labels.csv", r"V:\3D4EcoTec\down", img_trans = data_transform) 

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
        # print(las_name)
        views = sv.points_to_images(au.augment(las_name))
        
        # get side view
        views = torch.from_numpy(views[0:3,:,:])
        # views = Image.fromarray(views[0:3,:,:])
        
        # augment images
        if self.img_trans:
            views = self.img_trans(views)
        
        # return images with labels
        return {'views': views,
                'species': self.trees_frame.iloc[idx, 1]}

def collate_fn(list_items):
     x = torch.unsqueeze(list_items[0]["views"], dim = 0)
     y = torch.tensor([list_items[0]["species"]])
     for X in list_items[1:]:
          x = torch.cat((x, torch.unsqueeze(X["views"], dim = 0)), dim = 0)
          y = torch.cat((y, torch.tensor([X["species"]])), dim = 0)
          
     x = x.to("cuda", non_blocking=True, dtype=torch.float)
     y = y.to("cuda", non_blocking=True)
     return x, y
 
# found a more ellegant sollution need to test:
# def collator(batch):
#     X, Y = [], []
#     for x, y in batch:
#         X += [x, ]
#         Y += [y, ]
#     return torch.stack(X), torch.stack(Y)

#%%

# # load empty densenet201 model
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights = None)

# # create dataset object
tree_dataset = TrainDataset_testing(r"V:\3D4EcoTec\train_labels.csv", r"V:\3D4EcoTec\down")

# # create data loader
batch_size = 4
data_loader = torch.utils.data.DataLoader(tree_dataset, batch_size, collate_fn = collate_fn, shuffle = True)
images, labels = next(iter(data_loader))

#%%

# Define the transform to apply to the images
trafo = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])


# Define the dataset and data loader
dataset = TrainDataset_testing(r"V:\3D4EcoTec\train_labels.csv", r"D:\TLS\Puliti_Reference_Dataset\down")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, collate_fn = collate_fn) #num_workers=5 ,, pin_memory=True

images, labels = next(iter(dataloader))

# %%

# Define the model
model = torchvision.models.densenet201(weights='DenseNet201_Weights.DEFAULT')
# model.features[0]
# model.features[0] = torch.nn.Conv2d(1, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, 33)

# get the cuda device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

# give to
model.to(device)
# dataloader.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for i, data in enumerate(dataloader, 0):
        print(i)
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
# %%
from PIL import Image
from torchvision import transforms
input_image = Image.open(r"C:\Users\Julian\Desktop\Poster Auswahl\DSC10447.JPG")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
input_batch = input_batch.to('cuda')
with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])