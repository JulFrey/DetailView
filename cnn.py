# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:26:11 2023

@author: Julian
"""

# import packages
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
import torchvision
from torchvision import transforms
import datetime

# import own functions
import augmentation as au
import sideview as sv
import read_las as rl

# set  number of classess
n_class = 33
n_vali = 400

# set paths
path_csv_train = r"C:\Baumartenklassifizierung\train_labels.csv" #r"S:\3D4EcoTec\train_labels.csv"
path_csv_vali  = r"C:\Baumartenklassifizierung\vali_labels.csv" #r"S:\3D4EcoTec\vali_labels.csv"
path_las       = r"C:\Baumartenklassifizierung\data\down"

#%% setup new dataset class

# create dataset class to load the data from csv and las files
class TrainDataset_AllChannels():
    
    """Tree species dataset."""
    
    # initialization
    def __init__(self, csv_file, root_dir, img_trans = None, pc_rotate = True):
        
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
        self.pc_rotate   = pc_rotate
    
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
            image = sv.points_to_images(au.augment(las_name), res_im = 128)
        else:
            image = sv.points_to_images(rl.read_las(las_name), res_im = 128)
        image = torch.from_numpy(image)
        
        # # augment images (by channel)
        # if self.img_trans:
        #     new_image = torch.zeros_like(image)
        #     for c in range(0, image.shape[0]):
        #         new_image[c,:,:] = self.img_trans(image[c,:,:])
        #     image = new_image
        #     del new_image
        
        # augment images (all channels at once)
        if self.img_trans:
            image = self.img_trans(image)
        
        # add dimension
        image = image.unsqueeze(1)
        
        # get height
        height = torch.tensor(self.trees_frame.iloc[idx, 2], dtype = torch.float32)
        
        # get species
        label = torch.tensor(self.trees_frame.iloc[idx, 1], dtype = torch.int64)
        
        # return images with labels
        return image, height, label
    
    # training weights
    def weights(self):
        return torch.tensor(self.trees_frame["weight"].values)
        
#%% testing dataset & dataloader

# # setting up image augmentation
# img_trans = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()])

# # create dataset object
# dataset = TrainDataset_AllChannels(path_csv_train, path_las) # without
# dataset = TrainDataset_AllChannels(path_csv_train, path_las, img_trans = img_trans) # with

# # show image
# plt.imshow(dataset[0][0][0,0,:,:], interpolation = 'nearest')
# plt.show()

# # define a sampler
# train_size = 2**13
# sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights(), train_size, replacement = True)

# # create data loader
# batch_size = 2**3
# dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = sampler, pin_memory = True)

# # # test output of iterator
# # image, height, label = next(iter(dataloader))
# # print(image.shape); print(height.shape); print(label.shape)

#%% checking value distribution

# # create data loader
# batch_size_test = 100
# dataloader_test = torch.utils.data.DataLoader(dataset, batch_size = batch_size_test, sampler = sampler, pin_memory = True)

# # check value distribution
# image, height, label = next(iter(dataloader_test))
# plt.hist(height.numpy(), bins = 33); plt.show()
# plt.hist(label.numpy(), bins = 33); plt.show()

#%% setup simple view

# https://github.com/isaaccorley/simpleview-pytorch/blob/main/simpleview_pytorch/simpleview.py
class SimpleView(torch.nn.Module):

    def __init__(self, num_views: int, num_classes: int):
        super().__init__()
        
        # load backbone
        backbone = torchvision.models.densenet201(weights = "DenseNet201_Weights.DEFAULT")
        
        # change first layer to greyscale
        backbone.features[0].in_channels = 1
        backbone.features[0].weight = torch.nn.Parameter(backbone.features[0].weight.sum(dim = 1, keepdim = True))
        
        # remove effect of classifier
        z_dim = backbone.classifier.in_features
        backbone.classifier = torch.nn.Identity()
        
        # add new classifier
        self.backbone = backbone
        self.classifier = torch.nn.Linear(
            in_features = z_dim * num_views,
            out_features = num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = x.shape
        x = x.reshape(b * v, c, h, w) # batch * views
        z = self.backbone(x)
        z = z.reshape(b, v, -1)
        z = z.reshape(b, -1)
        return self.classifier(z)

#%% prepare simple view

# setting up image augmentation
img_trans = transforms.Compose([
    transforms.RandomHorizontalFlip(0.25),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees = 10, translate = (0.25, 0.25), scale = (0.75, 1.25))
    ])

# prepare data
dataset = TrainDataset_AllChannels(path_csv_train, path_las, img_trans = img_trans)

# define a sampler
train_size = 2**13
sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.weights(), train_size, replacement = True)

# create data loader
batch_size = 2**3
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = sampler, pin_memory = True)

# load the model
model = SimpleView(num_views = 7, num_classes = n_class)

# get the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

# give to device
model.to(device)

# # get test prediction
# inputs, heights, labels = next(iter(dataloader))
# inputs, labels = inputs.to(device), labels.to(device)
# preds = model(inputs)

# define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 5, verbose = True)

#%% training loop

# prepare validation data for checking
vali_dataset = TrainDataset_AllChannels(path_csv_vali, path_las)
vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size = 2**3, shuffle = True, pin_memory = True)

# prepare training
num_epochs = 100
best_v_loss = 1000
last_improvement = 0
timestamp = datetime.datetime.now().strftime('%Y%m%H%M')

# loop through epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    
    # loop through whole dataset?
    for i, data in enumerate(dataloader, 0): 
        
        # print progress every ten batches
        if i % 10 == 9:
            progress = (i / len(dataloader))
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
        
        # print loss every 100 batches
        running_loss += loss.item()
        if i % 100 == 99:
            print('[epoch: %d, batch: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
            
        # clear memory
        del inputs, height, labels
        torch.cuda.empty_cache()
            
    # validation loss
    running_v_loss = 0
    model.eval()
    for j, v_data in enumerate(vali_dataloader, 0):
        v_inputs, v_heights, v_labels = next(iter(vali_dataloader))
        v_inputs, v_labels = v_inputs.to("cuda"), v_labels.to("cuda")
        v_outputs = model(v_inputs)
        v_loss = criterion(v_outputs, v_labels)
        running_v_loss += v_loss.item()
        del v_inputs, v_heights, v_labels
        torch.cuda.empty_cache()
    avg_v_loss = running_v_loss / len(vali_dataloader)
    model.train()
    print('[epoch: %d] validation loss: %.3f' %
          (epoch + 1, avg_v_loss))
    
    # adjust learning rate
    scheduler.step(avg_v_loss)
    
    # save best model
    if avg_v_loss < best_v_loss:
        best_v_loss = avg_v_loss
        model_path = "model_{}_{}".format(timestamp, epoch + 1)
        torch.save(model.state_dict(), model_path)
        last_improvement = 0
    else:
        last_improvement += 1
    
    # check how long last improvement was ago
    if last_improvement > 10:
        break

torch.cuda.empty_cache()
print('\nFinished training\n')

#%% validating cnn

# load best model
model = SimpleView(num_views = 7, num_classes = n_class)
model.load_state_dict(torch.load("model_2023052202_9"))

# get the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

# give to device
model.to(device)

# turn on evaluation mode
model.eval()

# prepare data for validation
vali_dataset = TrainDataset_AllChannels(path_csv_vali, path_las)
vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size = 2**3, pin_memory = True)

# create metrics
accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = int(n_class)).to(device)
f1 = torchmetrics.F1Score(task = "multiclass", num_classes = int(n_class)).to(device)

# iterate over validation dataloader in batches
for data in vali_dataloader:
    v_inputs, v_heights, v_labels = data
    v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

    # get predictions
    v_preds = model(v_inputs)

    # calculate metrics for the batch
    accuracy.update(v_preds, v_labels)
    f1.update(v_preds, v_labels)

# get the final metrics
final_accuracy = accuracy.compute()
final_f1 = f1.compute()

# print final metrics
print('accuracy: %.3f' % final_accuracy)
print('f1: %.3f' % final_f1)
