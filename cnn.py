# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:26:11 2023

@author: Julian
"""

# import packages
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torchmetrics
import torchvision
from torchvision import transforms
import datetime

# import own functions
import augmentation as au
import sideview as sv

# set  number of classess
n_class = 33
n_vali = 400

# set paths
path_csv_train = r"S:\3D4EcoTec\train_labels.csv"
path_csv_vali  = r"S:\3D4EcoTec\vali_labels.csv"
path_las       = r"C:\Baumartenklassifizierung\data\down"

#%% setup new dataset class

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
        image = sv.points_to_images(au.augment(las_name)) # turn off for validation?
        
        # get side views
        image = torch.from_numpy(image)
        
        # augment images
        if self.img_trans:
            image = self.img_trans(image)
        
        # get height
        height = torch.tensor(self.trees_frame.iloc[idx, 2], dtype = torch.float32)
        
        # get species
        label = torch.tensor(self.trees_frame.iloc[idx, 1], dtype = torch.int64)
        
        # return images with labels
        return image, height, label
    
    # training weights
    def weights(self):
        return torch.tensor(self.trees_frame["weight"].values)

#%% setup new dataset class for testing

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
        image = sv.points_to_images(au.augment(las_name)) # turn off for validation?
        
        # get selected side & top views
        image = torch.from_numpy(image[self.channel,:,:])
        image = torch.unsqueeze(image, dim = 0)
        
        # augment images
        if self.img_trans:
            image = self.img_trans(image)
        
        # get height
        height = torch.tensor(self.trees_frame.iloc[idx, 2], dtype = torch.float32)
        
        # get species
        label = torch.tensor(self.trees_frame.iloc[idx, 1], dtype = torch.int64)
        
        # return images with labels
        return image, height, label
    
    # training weights
    def weights(self):
        return torch.tensor(self.trees_frame["weight"].values)
        
#%% setup dataset & dataloader

# setting up image augmentation
trafo = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(64), # for testing the model
    transforms.ToTensor()
])

# create dataset object
# dataset = TrainDataset_SingleChannel(path_csv_train, path_las, img_trans = trafo) # without
dataset = TrainDataset_SingleChannel(path_csv_train, path_las, img_trans = trafo) # with

# # show image
# plt.imshow(dataset[0][0][0,:,:], interpolation = 'nearest')
# plt.show()

# define a sampler
train_size = 2**9 # 2**13 # TODO: change back, this was for profiling
sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights(), train_size, replacement = True)

# create data loader
batch_size = 2**4
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = sampler) # pin_memory = True, num_workers = 4

# # test output of iterator
# image, height, label = next(iter(dataloader))
# print(image.shape); print(height.shape); print(label.shape)

#%% checking value distribution

# # create data loader
# batch_size_test = 100
# dataloader_test = torch.utils.data.DataLoader(dataset, batch_size = batch_size_test, sampler = sampler) # pin_memory = True, num_workers = 4

# # check value distribution
# image, height, label = next(iter(dataloader_test))
# plt.hist(height.numpy(), bins = 33); plt.show()
# plt.hist(label.numpy(), bins = 33); plt.show()

#%% setup model

# load the model
model = torchvision.models.densenet201()

# change first layer
model.features[0] = torch.nn.Conv2d(1, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)

# change last layer
model.classifier = torch.nn.Linear(model.classifier.in_features, int(n_class)) # TODO

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
criterion = torch.nn.CrossEntropyLoss() # laut dem simpleview paper vllt smooth loss?
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

#%% training loop

# # prepare validation data for checking
# vali_dataset = TrainDataset_SingleChannel(path_csv_vali, path_las)
# vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size = 2**4, shuffle = True) # pin_memory = True, num_workers = 4

# loop through epochs
num_epochs = 1
best_v_loss = 1000
timestamp = datetime.datetime.now().strftime('%Y%m%H%M')
for epoch in range(num_epochs):
    
    #  model.train() # ich check pytorch nicht, aber sollte das nicht irgendwo auftauchen?
    running_loss = 0.0
    
    # loop through whole dataset?
    for i, data in enumerate(dataloader, 0): 
        
        # print progress every ten batches
        if i % 10 == 9:
            progress = (i / (len(dataset) / batch_size))
            print('[epoch: %d] dataset: %.2f%%' % (epoch + 1, progress * 100))
        
        # load data
        inputs, height, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
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
            
            # loss
            print('[epoch: %d, batch: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
            
        # clear memory
        del inputs, height, labels
        torch.cuda.empty_cache()
            
    # # validation loss
    # running_v_loss = 0
    # model.train(False)
    # for j, v_data in enumerate(vali_dataloader, 0):
    #     v_inputs, v_heights, v_labels = next(iter(vali_dataloader))
    #     v_inputs, v_labels = v_inputs.to("cuda"), v_labels.to("cuda")
    #     v_outputs = model(v_inputs)
    #     v_loss = criterion(v_outputs, v_labels)
    #     running_v_loss += v_loss.item()
    #     del v_inputs, v_heights, v_labels
    #     torch.cuda.empty_cache()
    # avg_v_loss = running_v_loss / len(vali_dataloader)
    # model.train(True)
    # print('[epoch: %d] validation loss: %.3f' %
    #       (epoch + 1, avg_v_loss))
    
    # # save best model
    # if avg_v_loss < best_v_loss:
    #     best_v_loss = avg_v_loss
    #     model_path = "model_{}_{}".format(timestamp, epoch + 1)
    #     torch.save(model.state_dict(), model_path)

print('Finished training')

#%% validating cnn

# # prepare data for validation
# vali_dataset = TrainDataset_SingleChannel(path_csv_vali, path_las)
# vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size = n_class) # pin_memory = True, num_workers = 4

# # get predictions
# v_inputs, v_heights, v_labels = next(iter(vali_dataloader))
# v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
# v_preds = model(v_inputs)

# # get accuracy
# accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = int(n_class)).to("cuda")
# print('accuracy: %.3f' % accuracy(v_preds, v_labels))

# # get f1 score
# f1 = torchmetrics.F1Score(task = "multiclass", num_classes = int(n_class)).to("cuda")
# print('f1 score: %.3f' % f1(v_preds, v_labels))
